#!/usr/bin/env python3
"""executor_node; push executor as an ExecutePush action server.
One macro-step push = one long-running, preemptable action goal.
The node works in two phases: a) align, where the arm aligns with the force direction,
and b) approach and push, where the object is pushed at loop rate of 30 to 50 Hz.
For testing purposes, the node can run be run using python arguments which overrides 
the ROS private params, e.g. `_exec/dry_run:=true _viz/arrow_len:=0.5`.

Goals name the contact point either by `loc_idx` (>=0, resolved here against the
object model at `~npz_path`, like the sim) or by an explicit `push_point` (loc_idx < 0).

Run example:
    # 1) Make sure an estop script is running in another terminal (estop SDK example), then:
    rosrun npm_control executor_node.py _exec/dry_run:=false _exec/hostname:=ROBOT_IP \
        _npz_path:=/path/to/object.npz
    # drive it using test scripts:
    rosrun npm_control test_push_action_static.py --force-mag 30    # bench (no object)
    rosrun npm_control test_push_action_track.py  --force-mag 30    # object track
"""
import os
import sys
import time

import numpy as np
import rospy
import actionlib
import tf2_ros
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA, Float32, String, UInt8
from visualization_msgs.msg import Marker
from npm_msgs.msg import (ExecutePushAction, ExecutePushFeedback, ExecutePushResult)

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.time_sync
import bosdyn.client.util
from bosdyn.api import geometry_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.frame_helpers import (ODOM_FRAME_NAME, VISION_FRAME_NAME,
                                         HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME,
                                         WR1_FRAME_NAME, get_a_tform_b)
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, block_for_trajectory_cmd,
                                         blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

# Add the package srcs so this runs even without the workspace devel sourced.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
# obs_lib is pure numpy/scipy. Do NOT reach for npm_policy.policy_lib here: it
# imports torch at module scope, which the executor process has no business loading.
sys.path.insert(0, os.path.join(_HERE, "..", "..", "npm_policy", "src"))
from npm_control import executor_lib as el
from npm_policy import obs_lib as ol



_PHASE_ENUM = {"ALIGN": ExecutePushFeedback.PHASE_ALIGN,
               "APPROACH": ExecutePushFeedback.PHASE_APPROACH,
               "PUSH": ExecutePushFeedback.PHASE_PUSH,
               "RETREAT": ExecutePushFeedback.PHASE_RETREAT,
               "TOPPLE": ExecutePushFeedback.PHASE_TOPPLE}


def _ee_force_vec_from_state(st):
    # Estimated end-effector force as a hand-frame Vec3 (N), or None if unavailable.
    # Vector rather than magnitude because the free-space bias zero is a vector:
    # subtracting magnitudes would leave the lateral bias in. Takes an
    # already-fetched RobotState so the force describes the same instant as the
    # transforms snapshot the caller reads its geometry from.
    man = st.manipulator_state
    if not man.HasField("estimated_end_effector_force_in_hand"):
        return None
    f = man.estimated_end_effector_force_in_hand
    return np.array([f.x, f.y, f.z], dtype=np.float64)


class ExecutorNodeReal(object):
    def __init__(self):
        rf = rospy.get_param("~exec/root_frame", "odom")
        if rf not in ("odom", "vision"):
            raise ValueError("~exec/root_frame must be 'odom' or 'vision', got %r" % rf)
        self.root_frame = {"odom": ODOM_FRAME_NAME, "vision": VISION_FRAME_NAME}[rf]
        # The calibrated LINK frame, not the raw mocap body: push_point arrives in link
        # coords, so tracking it in the raw Motive pivot frame misplaces it.
        self.object_frame = rospy.get_param("~exec/object_frame", "object_link")
        # Object model cloud, the same .npz the policy trains and infers against.
        # It makes goal.loc_idx executable: the sim's action names a point INDEX and
        # derives the contact from the model, so the executor must too.
        self.npz_path = rospy.get_param("~npz_path", "")
        # Contact-cloud margin; must match policy_node's, and _model_cb checks it.
        self.pcl_shrink = float(rospy.get_param("~pcl_shrink", 1.0))
        # Tolerance (m) on the loc_idx-vs-push_point agreement check below.
        self.push_point_tol = rospy.get_param("~exec/push_point_tol", 0.005)
        self.link_pts = None
        # Object-frame contact point latched by _resolve_push_point for the goal in
        # flight, and which of the two paths produced it.
        self._push_point = None
        self._point_src = "none"
        # Free-space EE force offset (hand-frame Vec3, N), re-measured at the
        # standoff for every goal because it drifts with arm pose and temperature.
        self._force_bias = np.zeros(3, dtype=np.float64)
        self.dry_run = rospy.get_param("~exec/dry_run", True)
        self.force_clip = rospy.get_param("~exec/force_clip", el.FORCE_NORM_CLIP)
        self.watchdog_margin = rospy.get_param("~exec/watchdog_margin", 15.0)
        self.arrow_len = rospy.get_param("~viz/arrow_len", 1.0)
        self.loop_rate = rospy.get_param("~exec/loop_rate", 40.0)
        self.cmd_horizon = rospy.get_param("~exec/cmd_horizon", 0.6)
        self.tf_timeout = rospy.get_param("~exec/tf_timeout", 0.2)
        # Lifetime (s) stamped on every SE2 base goal. The bosdyn client only fills
        # se2_trajectory_request.end_time when robot_command(end_time_secs=...) is
        # given; without it the goal arrives already expired and the base never plans
        # a step. Must outlive the reissue period (~retreat/retract_cmd_period) so the
        # goal does not lapse between reissues.
        self.se2_end_time = rospy.get_param("~exec/se2_end_time", 2.0)
        # for testing. Waits for N seconds at phase boundaries so transitions are visible
        self.phase_pause = rospy.get_param("~exec/phase_pause", 0.0)
        # verbose per-phase/subphase prints (distinct from bosdyn SDK logging)
        self.verbose = rospy.get_param("~exec/verbose", True)
        # block for Enter at phase gates (bench testing only)
        self.interactive = rospy.get_param("~exec/interactive", False)
        # Pusher-tip offset from the wrist frame (arm_link_wr1) along the tool x/z.
        # The ArmCartesianCommand controls the wrist by default; without this the
        # tip overshoots the contact by ~this much. Mocap ground truth ~0.24m x.
        self.tool_tip_x = rospy.get_param("~exec/tool_tip_x", 0.24)  # m, wr1->tip x
        self.tool_tip_z = rospy.get_param("~exec/tool_tip_z", 0.03)  # m, wr1->tip z
        self._tool_offset = (self.tool_tip_x, self.tool_tip_z)
        # tip relative to the `hand` frame (TF-reported), for distance checks that
        # read the hand frame: hand sits _SPOT_ARM_TOOL_OFFSET ahead of wr1.
        self._tip_from_hand = np.array(
            [self.tool_tip_x - el._SPOT_ARM_TOOL_OFFSET[0], 0.0,
             self.tool_tip_z - el._SPOT_ARM_TOOL_OFFSET[2]], dtype=np.float64)
        # time-sync establish: per-attempt timeout (s) and retry count
        self.time_sync_timeout = rospy.get_param("~exec/time_sync_timeout", 15.0)
        self.time_sync_retries = rospy.get_param("~exec/time_sync_retries", 3)
        # env-level (npm.yaml), flat key ~max_finger_reach, not namespaced
        self.max_finger_reach = rospy.get_param("~max_finger_reach", 0.6)

        # ALIGN
        # arm parks this far in front of the contact on the push axis during ALIGN (m)
        self.align_standoff = rospy.get_param("~align/align_standoff", 0.15)
        # base standoff distance behind contact (m) during ALIGN phase;
        self.base_standoff = rospy.get_param("~align/base_standoff", 0.73)
        self.base_walk_timeout = rospy.get_param("~align/base_walk_timeout", 10.0)
        # SE2 velocity limit for the base pre-walk so alignment is not abrupt.
        self.base_max_lin_vel = rospy.get_param("~align/base_max_lin_vel", 0.5)  # m/s
        self.base_max_ang_vel = rospy.get_param("~align/base_max_ang_vel", 0.75)  # rad/s
        # Body-frame carry pose the arm tucks to during the base pre-walk, so the
        # extended hand does not sweep the object while the body advances (no
        # object pointcloud available at this stage to plan arm collisions).
        self.carry_hand_x = rospy.get_param("~align/carry_hand_x", 0.47)  # m, body-frame
        self.carry_hand_z = rospy.get_param("~align/carry_hand_z", 0.41)  # m, body-frame
        self.carry_move_time = rospy.get_param("~align/carry_move_time", 2.5)  # s
        # Rate caps on that retract. The move starts from the extended arm_ready
        # pose, close enough to elbow-straight that an uncapped Cartesian command
        # asks for a large joint velocity on the first ticks; the accel cap is what
        # removes the spike. None on either leaves the arm default (fast).
        self.carry_max_lin_vel = rospy.get_param("~align/carry_max_lin_vel", 0.25)  # m/s
        self.carry_max_accel = rospy.get_param("~align/carry_max_accel", 0.5)  # m/s^2
        # Base heading rule for the pre-walk.
        #   "pivot"    aim the body from the contact at the object's yaw pivot, so
        #              the assist's leg drive has no moment about the object's
        #              vertical axis (see el.aim_dir_to_pivot).
        #   "push_dir" aim down the policy force, the behaviour before this existed.
        # Off-axis contacts are exactly the ones that yaw the object away from the
        # crawl, so "pivot" is the default.
        self.aim_mode = rospy.get_param("~align/aim_mode", "pivot")
        # How far (deg) the aim may depart from the push axis. Past this the crawl is
        # driving the object somewhere the policy did not pick, and the standoff pose
        # (built along push_dir) sits align_standoff*sin(cone) off the body x axis.
        self.aim_cone = np.deg2rad(rospy.get_param("~align/aim_cone", 35.0))
        # Height band (m) above the object's lowest point that counts as its support
        # polygon. Wide enough to catch a real footprint, narrow enough that a tipping
        # object gives only the edge it is rotating over.
        self.aim_support_eps = rospy.get_param("~align/support_eps", 0.015)
        # Contact-to-pivot distance (m) below which the aim is meaningless (and the
        # moment it corrects is already ~zero): fall back to the push axis.
        self.aim_min_lever = rospy.get_param("~align/min_lever", 0.05)

        # APPROACH
        # The approach is an all-position, speed-limited drive, NOT a small force
        # command. A force-mode approach has nothing to react against across the
        # standoff, so it accelerates the whole way and lands as an impact: bag
        # spot_push_policy_6 logs 24-66 N arrival peaks off a 5 N command, and the
        # peak_force those goals reported was the impact, not the push.
        self.approach_vel = rospy.get_param("~approach/approach_vel", 0.05)      # m/s
        self.approach_accel = rospy.get_param("~approach/approach_accel", 0.25)  # m/s^2
        # Target sits this far PAST the contact along the push axis, so the position
        # controller is still commanding motion when the tip meets the surface.
        self.approach_overshoot = rospy.get_param("~approach/approach_overshoot", 0.02)
        # No-contact bound on the approach (s). Must outlast align_standoff /
        # approach_vel with margin, or a legitimate slow approach reads as a miss.
        self.approach_timeout = rospy.get_param("~approach/approach_timeout", 8.0)
        # Seconds of stationary, free-space force samples averaged at the standoff
        # and subtracted as the zero. The EE force is a torque-derived estimate with
        # a standing tool/model bias of a few N; without this the bias eats the
        # contact_made_n margin now that arrival is gentle instead of an impact.
        self.bias_window = rospy.get_param("~approach/bias_window", 0.3)
        # Second contact signal, independent of the force estimate and available
        # only because the approach is position-controlled: a tip commanded at
        # approach_vel that stops advancing has been blocked by something. It
        # covers the case the force gauge cannot, a contact whose load stays under
        # contact_made_n (light object, bad bias). Computed from TF, so no extra
        # per-tick RPC; the arm's own tracking-error feedback would need one and
        # would lag a command that is reissued every tick anyway.
        self.approach_stall_window = rospy.get_param("~approach/stall_window", 0.5)
        # Tip advance (m) over that window below which the approach reads as blocked.
        # Free-space travel over the window is approach_vel * stall_window (25 mm at
        # the defaults), so this sits well clear of a healthy approach.
        self.approach_stall_eps = rospy.get_param("~approach/stall_eps", 0.008)
        # Watchdog ceiling (N) for the approach alone. The push watchdog is
        # force_clip + watchdog_margin because the push commands force_clip; the
        # approach commands NO force, so any large load there is an anomaly (a
        # missed contact latch driving a stiff position move into the object) and
        # 90 N would be no margin at all.
        self.approach_force_ceiling = rospy.get_param("~approach/force_ceiling", 30.0)

        # PUSH
        self.contact_eps = rospy.get_param("~push/contact_eps", 2.0)
        self.contact_made_n = rospy.get_param("~push/contact_made_n", 5.0)
        if rospy.has_param("~push/approach_force"):
            # The approach no longer commands a force at all, so a config still
            # setting this is describing behaviour that is gone.
            rospy.logwarn("~push/approach_force is OBSOLETE and IGNORED; the approach "
                          "is now a speed-limited position move (~approach/approach_vel). "
                          "Use ~push/push_force_start to seed the push ramp.")
        # Force (N) the push ramp starts from. Defaults to contact_made_n, the force
        # the tip is already carrying at the instant contact latches: seeding at 0
        # hands a limp push axis to an object that is touching it, and for a small
        # goal force the ramp can take longer to climb back over contact_eps than
        # contact_lost_grace allows, ending the push as a spurious contact_lost.
        # This is NOT the old approach_force - the tip is already at rest against the
        # surface here, so it cannot become an impact.
        self.push_force_start = rospy.get_param("~push/push_force_start",
                                                self.contact_made_n)
        # The EE force estimate must stay below contact_eps for this long (s) before a
        # push counts as contact_lost. A single sub-threshold tick is noise, not a lost
        # contact; it happens routinely at the approach->push ramp and at the
        # push->topple handover, and killing the push there is a real-world failure.
        self.contact_lost_grace = rospy.get_param("~push/contact_lost_grace", 0.4)
        # time over which the push force ramps from push_force_start to full (s)
        self.ramp_time = rospy.get_param("~push/ramp_time", 1.0)
        # NOTE: the base does NOT follow the hand during the push. It is planted for
        # the whole phase, so the push is a pure arm motion and the arm's own reach
        # (_reach_gauge) is the honest travel limit. The only base motion inside
        # a push is the body assist below.

        # BODY ASSIST (topple / reach)
        # Body-driven continuation of a push whose arm has run out of travel: the arm
        # locks rigid to the body and the base crawls forward, pushing with the legs
        # through the arm as a strut. Two modes share the whole mechanism and this
        # param group, and differ only in their exit conditions (see _body_assist):
        #   topple - the object is part-way through tipping over; finish the tip.
        #   reach  - the object is NOT tipping, the arm simply ran out of travel
        #            before the push reached max_finger_reach; finish the slide.

        # Root frame the tilt gauge measures in. NOT the push root: the push is
        # commanded in `odom`, but the odom <- object_link TF path runs through the
        # Spot driver (~12 Hz, ~0.09s mean stamp lag), and tf2 resolves Time(0) to the
        # LATEST COMMON time over the whole chain, so it drags the 100+ Hz mocap object
        # pose back onto the driver's clock. Bag spot_push_policy_3 fits a 0.14-0.52s
        # pure delay on the logged tilt; at the 40+ deg/s tip rates seen there that is
        # 6-22deg of tilt the detector never sees, which is how a 22deg tip logs 9.5deg
        # and misses a 10deg threshold. `world` reaches the object through static TFs
        # and the mocap bridge only, so it is fresh. Any gravity-aligned frame is valid:
        # tilt is a delta, and only the yaw/tip split depends on the root. Empty string
        # -> use the push root frame (the old, laggy behaviour).
        self.tilt_root_frame = rospy.get_param("~topple/tilt_root_frame", "world")
        # yaw-free object rotation since the push started that reads as "toppling".
        # The param is in DEGREES so it is tunable by eye on the robot; everything
        # downstream compares in radians.
        self.topple_tilt_min_deg = rospy.get_param("~topple/tilt_min_deg", 30.0)
        self.topple_tilt_min = np.deg2rad(self.topple_tilt_min_deg)
        if rospy.has_param("~topple/tilt_min"):
            # A silent unit change is how a 30deg threshold becomes a 0.52deg one.
            rospy.logwarn("~topple/tilt_min (radians) is DEPRECATED and IGNORED; using "
                          "~topple/tilt_min_deg=%.2fdeg (%.3frad). Remove the old key.",
                          self.topple_tilt_min_deg, self.topple_tilt_min)
        # enable the reach-mode assist (topple mode is always available)
        self.reach_assist = rospy.get_param("~topple/reach_assist", True)
        # forward body-frame crawl speed (m/s) during the assist
        self.topple_body_vel = rospy.get_param("~topple/body_vel", 0.15)
        # hard cap (m) on body displacement from the latch pose; bounds the assist
        self.topple_max_body_travel = rospy.get_param("~topple/max_body_travel", 0.5)
        # same cap for reach mode; additionally floored by the drift the push still
        # owes, so the body never walks further than max_finger_reach needs.
        self.reach_max_body_travel = rospy.get_param("~topple/reach_max_body_travel", 0.5)
        # window (s) over which tip advance along the push axis is measured for stall
        self.topple_stall_window = rospy.get_param("~topple/stall_window", 0.5)
        # tip advance (m) below which, over that window, the arm counts as stalled
        self.topple_stall_eps = rospy.get_param("~topple/stall_eps", 0.01)
        # grace (s) for the tip to re-load after the force->position handover before
        # the assist gives up on the contact
        self.topple_load_grace = rospy.get_param("~topple/load_grace", 1.0)
        # lifetime (s) stamped on each base velocity command; must outlive one tick
        self.topple_vel_end_time = rospy.get_param("~topple/vel_end_time", 0.5)
        # Let a push that is mid-topple outlive goal.duration. The macro-step clock is
        # sized for a slide; quitting on it halfway through a tip leaves the object
        # balanced on an edge, which is a worse state than either finishing or never
        # starting. Bounded by max_time_beyond_duration_override and by tilt actually
        # still growing, so a stalled or finished tip gives the clock straight back.
        self.topple_duration_override = rospy.get_param("~topple/duration_override", True)
        # hard cap (s) on how far past goal.duration the override can run
        self.topple_max_beyond_duration = rospy.get_param(
            "~topple/max_time_beyond_duration_override", 4.0)
        # tilt gain (deg) over one stall_window below which the tip counts as finished
        # or stuck, ending the override. DEGREES for the same reason as tilt_min_deg.
        self.topple_tilt_rise_eps_deg = rospy.get_param("~topple/tilt_rise_eps_deg", 0.5)
        self.topple_tilt_rise_eps = np.deg2rad(self.topple_tilt_rise_eps_deg)

        # RECONTACT: the short standing hold at the very head of a TOPPLE assist,
        # ahead of RECENTER. The arm-only push ends in FORCE mode with the tip still
        # travelling, so the pose every later stage holds has to be read AFTER the arm
        # has stopped - and re-read once the object has fallen back onto it. Topple
        # mode only; a slide never leaves the tool. See doc/recontact_hold.md.
        self.recontact = rospy.get_param("~topple/recontact", True)
        # Total budget (s): the object breaking contact as it tips away, PLUS the fall
        # back onto the tool. Bags spot_push_policy_8/9: contact is lost 0.04-0.99s
        # after the handover and comes back 0.5-2.1s after it.
        self.recontact_wait = rospy.get_param("~topple/recontact_wait", 2.5)
        # Continuously-loaded time (s) that ends the stage early when the object never
        # breaks away at all. Without it every push pays the full budget standing still.
        self.recontact_settle = rospy.get_param("~topple/recontact_settle", 0.5)

        # RECENTER: the base-catch-up stage that runs at the head of a TOPPLE assist,
        # before the arm is locked to the body. Topple mode only; reach mode still
        # locks at whatever extension the push ended in.
        self.recenter = rospy.get_param("~topple/recenter", True)
        # Tip reach (m, shoulder-radial) at which the arm is considered to have its
        # working envelope back and the crawl can start. The mirror of
        # ~retreat/max_tip_reach, which is what ENDS the arm-only push.
        self.recenter_target_reach = rospy.get_param("~topple/recenter_target_reach",
                                                     0.75)
        # same gate when ~retreat/reach_signal is "manipulability"
        self.recenter_manip_min = rospy.get_param("~topple/recenter_manip_min", 0.05)
        # Floor (m) on that reach: the base is walking INTO the arm here, so past
        # some point the arm is folded up against the body and no better off than it
        # was at full extension.
        self.recenter_min_reach = rospy.get_param("~topple/recenter_min_reach", 0.55)
        # Hard ceiling (m) on recenter body travel, and the fallback cap whenever the
        # live clearance is unavailable (no mesh, no object TF).
        self.recenter_max_travel_cap = rospy.get_param(
            "~topple/recenter_max_travel_cap", 0.30)
        self.recenter_timeout = rospy.get_param("~topple/recenter_timeout", 3.0)
        # Crawl speed (m/s) while the tip is held in world. Slower than body_vel: the
        # arm is folding through a whole IK re-solve rather than riding along.
        self.recenter_body_vel = rospy.get_param("~topple/recenter_body_vel", 0.10)
        # Keep steering the commanded tool orientation from the tracked push force
        # through the crawl as well, instead of freezing it at the recenter exit.
        # The object keeps tipping, so the force direction - and with it the pitch
        # that stays normal to the pushed face - keeps dropping.
        self.recenter_track_orientation = rospy.get_param(
            "~topple/recenter_track_orientation", True)
        # Bound (deg) on how far the commanded elevation may wander from the one the
        # push started with, and the slew limit (deg/s) on getting there. The
        # orientation is derived from a tracked object pose; a dropped or mis-solved
        # track must not be able to swing a loaded wrist.
        self.recenter_pitch_band_deg = rospy.get_param("~topple/recenter_pitch_band",
                                                       25.0)
        self.recenter_pitch_band = np.deg2rad(self.recenter_pitch_band_deg)
        self.recenter_pitch_rate_deg = rospy.get_param("~topple/recenter_pitch_rate",
                                                       30.0)
        self.recenter_pitch_rate = np.deg2rad(self.recenter_pitch_rate_deg)
        # Apply the same live clearance cap to the body-locked crawl. The crawl is
        # where the base actually closes on the object, so this is worth more there
        # than in recenter.
        self.recenter_clearance_gate = rospy.get_param(
            "~topple/recenter_clearance_gate", True)
        # AIM TRACKING. ALIGN points the body at the pivot before the push; the push
        # then yaws the object, so by the assist that heading is stale. These steer
        # the base back onto the live aim while it crawls.
        self.aim_track = rospy.get_param("~topple/aim_track", True)
        # Yaw rate ceiling (deg/s) and the proportional gain (1/s) on the aim error.
        self.aim_yaw_rate = np.deg2rad(rospy.get_param("~topple/aim_yaw_rate", 10.0))
        self.aim_gain = rospy.get_param("~topple/aim_gain", 1.0)
        # Error (deg) below which the base does not steer at all. The pivot comes from
        # a tracked pose and a sampled cloud; without a deadband the base hunts.
        self.aim_deadband = np.deg2rad(rospy.get_param("~topple/aim_deadband", 5.0))
        # Cumulative base yaw (deg) allowed during the BODY-LOCKED crawl. Every
        # radian there also carries the latched tip off the body x axis (the tip
        # counter-rotation keeps it out of the object, but it still spends arm
        # envelope), so the crawl gets a budget the recenter does not need.
        self.aim_yaw_max = np.deg2rad(rospy.get_param("~topple/aim_yaw_max", 20.0))

        # CLEARANCE: axis-aligned body box in flat_body, tested against the object's
        # collision cloud to decide how much further the base may crawl.
        self.clr_samples = int(rospy.get_param("~clearance/samples", 2000))
        # mesh-vs-npz bounding box disagreement (m) that warns at load
        self.clr_bbox_tol = rospy.get_param("~clearance/bbox_tol", 0.005)
        self.clr_half_len = rospy.get_param("~clearance/half_len", 0.55)
        self.clr_half_width = rospy.get_param("~clearance/half_width", 0.25)
        self.clr_half_height = rospy.get_param("~clearance/half_height", 0.10)
        # front knees swing ahead of the body while walking; y likewise
        self.clr_front_margin = rospy.get_param("~clearance/front_margin", 0.12)
        self.clr_side_margin = rospy.get_param("~clearance/side_margin", 0.05)
        # cloud above half_height + over_margin is an overhang to walk UNDER, and
        # below -(half_height + under_margin) passes under the belly; neither blocks
        self.clr_over_margin = rospy.get_param("~clearance/over_margin", 0.05)
        self.clr_under_margin = rospy.get_param("~clearance/under_margin", 0.05)
        # box/point marker publish rate (Hz), decoupled from the 40 Hz control loop
        self.clr_viz_rate = rospy.get_param("~clearance/viz_rate", 5.0)
        self._clr_viz_last = 0.0
        # The aim markers ride the same rate, on their own clock so the two throttles
        # do not steal ticks from each other.
        self._aim_viz_last = 0.0
        # Collision cloud in the object LINK frame, filled by _load_object_model.
        # None disables every clearance gate (and says so, loudly, at startup).
        self.collision_pts = None

        # RETREAT
        # Body-frame floor (m) for the pusher tip: the retract target is derived from
        # the object pose, so nothing else stops it walking back into the robot when
        # the object follows the tip. _clamp_target_to_body enforces this.
        self.min_hand_body_x = rospy.get_param("~retreat/min_hand_body_x", 0.55)
        # which near-singularity signal _reach_gauge uses:
        # "shoulder_radius" | "manipulability"
        self.reach_signal = rospy.get_param("~retreat/reach_signal", "shoulder_radius")
        # far-reach limit (m) as the RADIAL tip distance from the shoulder pivot, the
        # measure the published reach spec uses
        # (https://support.bostondynamics.com/s/article/How-Spot-Arm-Moves-151690).
        # Geometric max for this arm + pusher tip is 0.9906m (el.shoulder_reach);
        # stay under it, the arm goes near-singular before the hard limit.
        self.max_tip_reach = rospy.get_param("~retreat/max_tip_reach", 0.93)
        # Yoshikawa index below which the arm is treated as near-singular
        self.manip_min = rospy.get_param("~retreat/manip_min", 0.02)
        # force (N) the release ramp starts from before decaying to zero
        self.retreat_hold_force = rospy.get_param("~retreat/retreat_hold_force", 10.0)
        # seconds to linearly decay the held force to zero before the retract
        self.release_time = rospy.get_param("~retreat/release_time", 2.0)
        # tip-to-object clearance (m) that ends the arm retract
        self.safe_object_dist = rospy.get_param("~retreat/safe_object_dist", 0.25)
        # how far behind the live contact (m), on the push axis, the tip is targeted
        # during the retract. Must exceed safe_object_dist or the loop cannot converge.
        self.retreat_dist = rospy.get_param("~retreat/retreat_dist", 0.30)
        # trajectory time (s) of each reissued retract pose command
        self.retract_move_time = rospy.get_param("~retreat/retract_move_time", 1.0)
        # consecutive object-TF misses tolerated before the retract bails to carry
        self.object_tf_max_misses = rospy.get_param("~retreat/object_tf_max_misses", 20)
        # min interval (s) between base SE2 reissues during retract; keeps the 40 Hz
        # monitor loop from superseding the base step plan every tick (base would only
        # weight-shift, never walk). The retract SE2 goal is near-constant, so ~2-3 Hz
        # is ample.
        self.retract_cmd_period = rospy.get_param("~retreat/retract_cmd_period", 0.4)
        # overall wall-clock budget (s) for the whole coordinated retreat
        self.retreat_deadline = rospy.get_param("~retreat/retreat_deadline", 15.0)
        # SE2 velocity limit for the base while it backs up during the retract.
        self.retreat_base_max_lin_vel = rospy.get_param("~retreat/base_max_lin_vel", 0.5)
        self.retreat_base_max_ang_vel = rospy.get_param("~retreat/base_max_ang_vel", 0.75)

        self._load_object_model()
        # The policy resolves nothing now: goals name the contact by index only, so
        # the only guard against the two nodes holding different models is this
        # fingerprint. Latched, so it arrives whichever node starts first.
        self.model_checked = False
        rospy.Subscriber("/npm/model_info", String, self._model_cb, queue_size=1)
        # ... but only if something publishes it. Nothing does when the stack runs
        # without policy_node, and an unverified model is silent otherwise.
        self.model_timeout = float(rospy.get_param("~exec/model_info_timeout", 5.0))
        rospy.Timer(rospy.Duration(self.model_timeout), self._model_timeout,
                    oneshot=True)

        self.robot = None
        self.command_client = None
        self.state_client = None
        self.lease_client = None
        if not self.dry_run:
            self._connect()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # last (frame, pos) a marker was drawn at; anchors the phase text marker
        self._last_contact = None
        self.marker_pub = rospy.Publisher("/npm/push_marker", Marker, queue_size=1)
        # debug topics. per-tick EE force and latched phase enum
        self.force_pub = rospy.Publisher("/npm/ee_force", Float32, queue_size=1)
        self.phase_pub = rospy.Publisher("/npm/executor/phase", UInt8, queue_size=1,
                                         latch=True)
        # Why an assist did or did not fire, one key=value line per push tick. The
        # topple/reach predicate ANDs several terms; without this the only symptom of a
        # mistuned threshold is a push that quietly ends in "arm_limit".
        self.assist_pub = rospy.Publisher("/npm/executor/assist_debug", String,
                                          queue_size=1)
        # Body box and the object points that block it, for rviz. The box is the
        # only part of the clearance test with no other observable: a number in a
        # log line does not show WHICH end of the object is in the way.
        self.body_box_pub = rospy.Publisher("/npm/executor/body_box", Marker,
                                            queue_size=2)
        self.clearance_pub = rospy.Publisher("/npm/executor/clearance_pts", Marker,
                                             queue_size=1)
        # Yaw pivot, the aim line the base is steered onto, and the push axis it is
        # being compared against. Three markers because the whole point is the ANGLE
        # between the last two; a heading number in a log line does not show it.
        self.aim_pub = rospy.Publisher("/npm/executor/aim", Marker, queue_size=3)
        # commanded arm standoff pose (position + orientation) for rviz inspection;
        # unlike the sphere marker this carries orientation (the fingertip target).
        self.standoff_pub = rospy.Publisher("/npm/standoff_pose", PoseStamped,
                                            queue_size=1, latch=True)
        # Start action server after power on sequence.
        # auto_start=False so we call start() at the end of run().
        self.server = actionlib.SimpleActionServer(
            "push", ExecutePushAction, execute_cb=self._execute_cb, auto_start=False)
        rospy.loginfo("executor_node up (dry_run=%s, root_frame=%s, object_frame=%s)",
                      self.dry_run, self.root_frame, self.object_frame)

    def _connect(self):
        bosdyn_verbose = rospy.get_param("~exec/bosdyn_verbose", False)
        hostname = rospy.get_param("~exec/hostname", None)
        if not hostname:
            raise ValueError("~exec/hostname is required when ~exec/dry_run is false")
        bosdyn.client.util.setup_logging(bosdyn_verbose)
        sdk = bosdyn.client.create_standard_sdk("NpmExecutorRealClient")
        self.robot = sdk.create_robot(hostname)
        bosdyn.client.util.authenticate(self.robot)
        # Time-sync wait/retry 
        self._wait_for_time_sync()

        assert self.robot.has_arm(), "Robot requires an arm to run this test."
        # For safety refuse to move without an external estop endpoint holding the line.
        assert not self.robot.is_estopped(), (
            "Robot is estopped. Register an external E-Stop (estop SDK example / "
            "GUI) before running this test.")

        self.command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name)
        self.state_client = self.robot.ensure_client(
            RobotStateClient.default_service_name)
        self.lease_client = self.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name)

    def _wait_for_time_sync(self):
        attempts = max(1, int(self.time_sync_retries))
        for i in range(1, attempts + 1):
            try:
                self._vlog("time_sync: establishing (attempt %d/%d, timeout %.1fs)",
                           i, attempts, self.time_sync_timeout)
                self.robot.time_sync.wait_for_sync(timeout_sec=self.time_sync_timeout)
                self._vlog("time_sync: synced.")
                return
            except bosdyn.client.time_sync.TimedOutError:
                if i >= attempts:
                    rospy.logerr("time_sync: failed after %d attempts.", attempts)
                    raise
                rospy.logwarn("time_sync: timed out (attempt %d/%d), retrying...",
                              i, attempts)

    def run(self):
        if self.dry_run:
            self.server.start()
            rospy.spin()
            return

        with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True,
                                                return_at_exit=True):
            self.robot.logger.info("Powering on...")
            self.robot.power_on(timeout_sec=20)
            assert self.robot.is_powered_on(), "Robot power on failed."

            self.robot.logger.info("Standing...")
            blocking_stand(self.command_client, timeout_sec=10)

            # Straight to the body-frame carry pose. arm_pose_command is accepted
            # from a stowed arm, so no arm_ready/unstow step is required first.
            self.robot.logger.info("Moving arm to carry pose...")
            self._go_to_carry_pose()

            rospy.on_shutdown(self._safe_stop)
            # Starting the action server after the robot is powered on, standing, and in carry pose.
            self.server.start()  
            self.robot.logger.info("Ready - ExecutePush action server up.")
            try:
                rospy.spin()
            finally:
                self._stow_and_power_off()

    def _load_object_model(self):
        # Cache the link-frame model cloud so loc_idx goals resolve without file IO.
        # Missing npz is not fatal: the push_point (bench) path still works, and a
        # loc_idx goal will be rejected explicitly rather than silently mis-aimed.
        if not self.npz_path:
            rospy.logwarn("~npz_path unset: loc_idx goals will be REJECTED; only "
                          "explicit push_point goals (loc_idx < 0) can run.")
            return
        try:
            pts, _normals, scale, centroid = ol.load_model_npz(self.npz_path)
        except Exception:
            rospy.logerr("Failed to load object model %s; loc_idx goals will be "
                         "rejected.", self.npz_path, exc_info=True)
            return
        self.link_pts = ol.link_frame_cloud(pts, scale, centroid, self.pcl_shrink)
        inset_mm = 1000.0 * ol.shrink_inset(pts, scale, self.pcl_shrink)
        rospy.loginfo("object model: %s (%d points, scale=%.4f pcl_shrink=%.4f -> "
                      "inset %.1f/%.1f/%.1f mm)",
                      self.npz_path, self.link_pts.shape[0], scale, self.pcl_shrink,
                      inset_mm[0], inset_mm[1], inset_mm[2])
        self._load_collision_cloud(pts, scale, centroid)

    def _load_collision_cloud(self, pts, scale, centroid):
        """
        Cache the object's surface cloud for the body-clearance test.

        The mesh path is the npz path with a .obj suffix, derived rather than
        configured: the two files are one asset, and a second parameter is one more
        thing that can be left pointing at the wrong mesh.

        The mesh is used VERBATIM as link-frame metres. The npz stores points
        normalised about the COM and link_frame_cloud undoes exactly that
        normalisation, so .obj coordinates and the link frame are the same thing.
        That is checked here against the npz bbox rather than assumed: a config.yaml
        with a non-unit scale, or an npz regenerated from a different mesh, would
        otherwise surface as Spot crawling into a box.

        pcl_shrink is deliberately NOT applied. It exists to keep the pointer argmax
        off physical edges; a collision test wants the real surface.

        Falls back to the un-shrunk npz cloud when the mesh is missing - 128 points
        that sit a fraction of a millimetre inside the true surface, which is a
        coarser test but still the right sign.
        """
        full_pts = ol.link_frame_cloud(pts, scale, centroid, 1.0)
        obj_path = os.path.splitext(self.npz_path)[0] + ".obj"
        if not os.path.isfile(obj_path):
            rospy.logerr("COLLISION MESH MISSING: %s not found. The body-clearance "
                         "test falls back to the %d-point npz cloud, which samples "
                         "the surface sparsely and can miss a face between points. "
                         "Put the .obj beside the .npz.",
                         obj_path, full_pts.shape[0])
            self.collision_pts = full_pts
            return
        try:
            cloud, n_verts, n_faces = el.load_mesh_cloud(obj_path,
                                                         samples=self.clr_samples)
        except Exception:
            rospy.logerr("Failed to parse collision mesh %s; falling back to the "
                         "npz cloud.", obj_path, exc_info=True)
            self.collision_pts = full_pts
            return
        lo = np.abs(cloud.min(axis=0) - full_pts.min(axis=0)).max()
        hi = np.abs(cloud.max(axis=0) - full_pts.max(axis=0)).max()
        if max(lo, hi) > self.clr_bbox_tol:
            rospy.logerr("COLLISION MESH MISMATCH: %s and %s disagree on the object "
                         "bounding box by %.1f mm (tol %.1f mm). The mesh is being "
                         "used as link-frame metres, so either config.yaml applies a "
                         "non-unit scale or the two files are different objects. "
                         "Clearance numbers are NOT trustworthy.",
                         obj_path, self.npz_path, 1000.0 * max(lo, hi),
                         1000.0 * self.clr_bbox_tol)
        self.collision_pts = cloud
        rospy.loginfo("collision mesh: %s (%d verts, %d tris -> %d points, bbox "
                      "agrees with the npz to %.1f mm)",
                      obj_path, n_verts, n_faces, cloud.shape[0],
                      1000.0 * max(lo, hi))

    def _model_cb(self, msg):
        if self.model_checked or not self.npz_path:
            return
        self.model_checked = True
        ok, text = ol.check_model_info(msg.data, self.npz_path, self.pcl_shrink)
        (rospy.loginfo if ok else rospy.logerr)(text)

    def _model_timeout(self, _event):
        # A latched publisher that never starts looks exactly like one that has not
        # started YET, so silence is the failure mode: nothing checks the model and
        # nothing says so. Warn once and keep going - a mismatch is an error, but an
        # unverified model is only unverified.
        if self.model_checked or not self.npz_path:
            return
        rospy.logwarn("no /npm/model_info after %.1f s: object model UNVERIFIED. "
                      "loc_idx goals still resolve, but nothing has confirmed the "
                      "publisher holds the same .npz and pcl_shrink=%.4f.",
                      self.model_timeout, self.pcl_shrink)

    def _resolve_push_point(self, goal):
        """
        Object-frame contact point for this goal, latched for its whole lifetime.

        loc_idx >= 0 is the policy/sim path: the executor derives the point from its
        own copy of the model, so the index and the model together define the contact
        exactly as objectmanip_env_discrete.step() does. loc_idx < 0 is the bench
        path, taking goal.push_point verbatim.

        Returns True on success, False after aborting the goal.
        """
        try:
            point, src = el.resolve_push_point(goal.loc_idx, goal.push_point,
                                               self.link_pts)
        except ValueError as exc:
            self._abort("error", str(exc))
            return False
        self._push_point, self._point_src = point, src
        if src == "loc_idx":
            # Bench rigs may still fill push_point. The coordinator does not, so this
            # is inert on the policy path; /npm/model_info carries the model check
            # there. Warn only: the executor's resolution is the authoritative one.
            sent = el.as_xyz(goal.push_point)
            if np.any(sent) and np.linalg.norm(sent - point) > self.push_point_tol:
                rospy.logwarn("push_point mismatch at loc=%d: goal=(%.3f,%.3f,%.3f) "
                              "vs model=(%.3f,%.3f,%.3f), %.4fm apart. Using the "
                              "model. Do policy and executor share ~npz_path?",
                              goal.loc_idx, sent[0], sent[1], sent[2],
                              point[0], point[1], point[2],
                              float(np.linalg.norm(sent - point)))
        return True

    def _object_pose_in_root(self, object_frame, root=None):
        # Look up the tracked object's pose in `root` (default: the push root frame).
        root = root or self.root_frame
        try:
            tf = self.tf_buffer.lookup_transform(root, object_frame,
                                                 rospy.Time(0),
                                                 rospy.Duration(self.tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "TF %s<-%s lookup failed: %s",
                                   root, object_frame, exc)
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return R, np.array([t.x, t.y, t.z])

    def _object_frame(self, goal):
        # Object TF frame for this goal; empty -> the executor's default
        # (--object-frame), mirroring root_frame.
        return goal.object_frame or self.object_frame

    def _tilt_rotation(self, object_frame, tilt_root):
        # Object rotation for the tilt gauge, in tilt_root. Deliberately a separate
        # lookup from _track's: the push root lags the object by the Spot driver's TF
        # latency (see ~topple/tilt_root_frame). None when the frame is unavailable.
        pose = self._object_pose_in_root(object_frame, root=tilt_root)
        return None if pose is None else pose[0]

    def _track(self, goal):
        """World contact + force for the goal, from the current object pose.

        The contact point is self._push_point, latched once per goal by
        _resolve_push_point.
        """
        pose = self._object_pose_in_root(self._object_frame(goal))
        if pose is None:
            return None
        R, t = pose
        force_body = [goal.force_body.x, goal.force_body.y, goal.force_body.z]
        return el.world_contact_from_object(R, t, self._push_point, force_body)

    def _execute_cb(self, goal):
        root = {"": self.root_frame, "odom": ODOM_FRAME_NAME,
                "vision": VISION_FRAME_NAME}.get(goal.root_frame, self.root_frame)
        if not self._resolve_push_point(goal):
            return
        track = self._track(goal)
        if track is None:
            self._abort("error", "object TF unavailable at goal start")
            return
        contact_world, force_world = track
        mag, q = el.task_frame_from_force(force_world)
        push_dir = force_world / max(np.linalg.norm(force_world), 1e-9)
        rospy.loginfo("GOAL loc=%d src=%s |F|=%.1fN root=%s point=(%.3f,%.3f,%.3f) "
                      "contact=(%.3f,%.3f,%.3f) dur=%.1fs%s",
                      goal.loc_idx, self._point_src, mag, root,
                      self._push_point[0], self._push_point[1], self._push_point[2],
                      contact_world[0], contact_world[1], contact_world[2],
                      goal.duration, " [dry_run]" if self.dry_run else "")
        self._publish_marker(root, contact_world, force_world, mag)
        # Draw the body box once per goal, before anything moves, so the geometry the
        # assist will gate on can be eyeballed against the object beforehand rather
        # than only mid-crawl. Published on the dry_run path too - previewing it is
        # most of what dry_run is for.
        self._clr_viz_last = 0.0
        self._publish_clearance_viz(*self._clearance(goal))
        # Same reason for the aim: which way the base will face, and how far that is
        # off the policy's push axis, is decided here and is worth seeing before the
        # robot walks anywhere.
        self._aim_viz_last = 0.0
        self._publish_aim_viz(root, contact_world, None,
                              self._pivot_in_root(goal, root), push_dir)

        if self.dry_run:
            res = ExecutePushResult(contact_made=False, peak_force=0.0,
                                    finger_travel=0.0, end_reason="dry_run")
            self.server.set_succeeded(res)
            return

        # align phase. base pre-walk such that the body sits behind the push point
        # before the arm extends
        self._phase_mark("ALIGN")
        self._publish_feedback(ExecutePushFeedback.PHASE_ALIGN, False, 0.0, 0.0)
        self._publish_sphere(root, contact_world, mid=1, rgba=(0.1, 1.0, 0.1, 0.9))
        # sub-phase: tuck the arm before the base moves. The hand is still at the
        # extended arm_ready pose and there is no object pointcloud yet to plan arm
        # collisions, so pull it into a body-frame carry pose that clears the object
        # as the body advances. The pose is body-relative, so it holds through the walk.
        self._vlog("ALIGN[arm]: retracting to carry pose (x=%.2f z=%.2f, body frame).",
                   self.carry_hand_x, self.carry_hand_z)
        try:
            self._go_to_carry_pose()
        except Exception:
            rospy.logerr("ALIGN carry retract failed - stopping arm, aborting goal.",
                         exc_info=True)
            self._stop_arm()
            self._abort("error", "align failed")
            return

        # sub-phase: base pre-walk
        self._vlog("ALIGN[base]: walking base behind contact along push_dir "
                   "(standoff=%.2fm).", self.base_standoff + self.align_standoff)
        if not self._walk_base_align(goal, contact_world, push_dir, root):
            self._stop_arm()
            self._abort("error", "align failed")
            return
        self._vlog("ALIGN[base]: base in position.")
        self._gate("base aligned -> arm cartesian align?")

        # sub-phase: arm-only standoff move; base already positioned so the target
        # is in reach. Park align_standoff in front of the contact on the push axis
        # so the approach into the object is a straight push-axis move (no lateral
        # sweep across the surface). approach_eps stays for the push loop's own use.
        self._vlog("ALIGN[arm]: moving hand to standoff %.2fm in front of contact "
                   "on push axis.", self.align_standoff)
        standoff_cmd, standoff = el.build_standoff_pose_command(
            contact_world, push_dir, q, root_frame=root, eps=self.align_standoff,
            tool_offset=self._tool_offset)
        self._publish_sphere(root, standoff, mid=2, rgba=(0.1, 0.4, 1.0, 0.9))
        self._publish_standoff_pose(root, standoff, q)
        try:
            sid = self.command_client.robot_command(standoff_cmd)
            block_until_arm_arrives(self.command_client, sid, 5.0)
        except Exception:
            rospy.logerr("ALIGN arm move failed - stopping arm, aborting goal.", exc_info=True)
            self._stop_arm()
            self._abort("error", "align failed")
            return
        self._vlog("ALIGN[arm]: hand at standoff, push-aligned.")
        self._gate("arm aligned -> approach + push?")

        self._phase_mark("APPROACH")
        # Zero the force estimate here: the arm is settled at the standoff and
        # provably touching nothing, which is the only moment that holds.
        self._measure_force_bias()
        # The task orientation is latched from ALIGN, not recomputed per tick: the
        # roll of task_frame_from_force comes from a Gram-Schmidt against an
        # arbitrary reference and is not continuous in the object pose, and the
        # approach holds every rotational axis in position mode, so a per-tick q
        # would spin the wrist on the way in.
        approach_reason, latch, approach_peak = self._approach_loop(goal, root, q)
        if approach_reason == "error":
            # Same stance as a push exception: stop, report, do not retry.
            self._abort("error", "approach exception", peak=approach_peak,
                        contact_made=latch.made)
            rospy.signal_shutdown("approach exception")
            return
        self._push_loop(goal, root, latch, approach_peak, approach_reason)

    def _measure_force_bias(self):
        # Average the free-space EE force with the arm parked at the standoff, and
        # subtract it as the contact gauge's zero. The estimate is torque-derived
        # and carries a standing tool/model offset of a few N; that offset used to
        # be swamped by the approach impact, but a speed-limited approach arrives
        # gently and the contact threshold now has to resolve real load.
        self._force_bias = np.zeros(3, dtype=np.float64)
        if self.bias_window <= 0.0:
            return
        samples = []
        rate = rospy.Rate(self.loop_rate)
        t_end = time.time() + self.bias_window
        while time.time() < t_end and not rospy.is_shutdown():
            v = _ee_force_vec_from_state(self.state_client.get_robot_state())
            if v is not None:
                samples.append(v)
            rate.sleep()
        if not samples:
            rospy.logwarn("APPROACH: no EE force samples for the bias zero; "
                          "thresholding on the raw estimate.")
            return
        self._force_bias = np.mean(samples, axis=0)
        self._vlog("APPROACH: force zero = (%.2f,%.2f,%.2f)N |%.2f|N over %d samples.",
                   self._force_bias[0], self._force_bias[1], self._force_bias[2],
                   float(np.linalg.norm(self._force_bias)), len(samples))

    def _contact_force(self, st):
        # (load, raw) EE force magnitudes in N, either None when unavailable.
        # `load` has the free-space bias removed and drives every contact decision;
        # `raw` is the unmodified estimate and drives the watchdog and peak_force,
        # which are about the true load on the arm, not about detecting a touch.
        v = _ee_force_vec_from_state(st)
        if v is None:
            return None, None
        return (float(np.linalg.norm(v - self._force_bias)),
                float(np.linalg.norm(v)))

    def _approach_loop(self, goal, root, task_quat):
        """
        Drive the pusher tip into the object at a bounded speed, in all-position
        mode with NO commanded force, until contact latches.

        Returns (end_reason, latch, peak). end_reason is None once contact is made
        and the push may run; any other value is terminal.
        """
        latch = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                self.contact_lost_grace)
        peak = 0.0
        # Approach commands NO force, so the push watchdog (force_clip + margin, 90 N)
        # is no margin at all here: any large load is an anomaly, not a strong push.
        ceiling = self.approach_force_ceiling
        # Time before the blocked-tip gauge is trusted: a full stall window of samples,
        # taken after the tip is up to speed (approach_vel / approach_accel).
        stall_gate = (self.approach_stall_window
                      + self.approach_vel / max(self.approach_accel, 1e-6))
        rate = rospy.Rate(self.loop_rate)
        # An arm-only command does not cancel the ALIGN pre-walk goal, so the stand
        # has to ride along with every arm command (same reason as in _push_loop).
        stand_mob = el.build_stand_mobility_command()
        tip_hist = []
        t_start = time.time()

        try:
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    return "preempted", latch, peak

                st = self.state_client.get_robot_state()
                snap = st.kinematic_state.transforms_snapshot

                track = self._track(goal)
                if track is not None:
                    contact_world, force_world = track
                    push_dir = force_world / max(np.linalg.norm(force_world), 1e-9)
                    # Re-derived every tick: the object frame moves, so the surface
                    # the approach is aiming at moves with it.
                    cmd, _ = el.build_approach_pose_command(
                        contact_world, push_dir, task_quat, root_frame=root,
                        overshoot=self.approach_overshoot, duration_s=self.cmd_horizon,
                        tool_offset=self._tool_offset, mobility_command=stand_mob,
                        max_lin_vel=self.approach_vel, max_accel=self.approach_accel)
                    self.command_client.robot_command(cmd)
                    root_T_hand = get_a_tform_b(snap, root, HAND_FRAME_NAME)
                    tip_now = np.array(root_T_hand.transform_point(*self._tip_from_hand))
                    self._sample_series(tip_hist, float(np.dot(tip_now, push_dir)),
                                        window=self.approach_stall_window)

                load, raw = self._contact_force(st)
                cur = 0.0 if raw is None else raw
                self.force_pub.publish(Float32(cur))
                latch.update(load, time.time())
                if raw is not None:
                    peak = max(peak, raw)
                    if raw > ceiling:
                        rospy.logerr("WATCHDOG: EE force %.1fN > %.1fN during approach "
                                     "- stopping arm.", raw, ceiling)
                        self._stop_arm()
                        return "watchdog", latch, peak

                self._publish_feedback(ExecutePushFeedback.PHASE_APPROACH,
                                       latch.made, cur, 0.0)
                if latch.made:
                    self._vlog("APPROACH: contact at load=%.1fN (raw=%.1fN) after "
                               "%.2fs, peak=%.1fN.", 0.0 if load is None else load,
                               cur, time.time() - t_start, peak)
                    return None, latch, peak

                # Blocked-tip fallback: a tip commanded at approach_vel that has
                # stopped advancing is touching something, whatever the force
                # estimate says. Only trusted once the move is up to speed, since
                # the tip is accelerating from rest for the first window.
                stalled, adv = self._tip_stalled(tip_hist,
                                                 window=self.approach_stall_window,
                                                 eps=self.approach_stall_eps)
                if stalled and (time.time() - t_start) > stall_gate:
                    latch.force_made(time.time())
                    rospy.logwarn("APPROACH: tip blocked (advance %.4fm < %.4fm over "
                                  "%.2fs) with load %.1fN below contact_made_n %.1fN; "
                                  "latching contact anyway. Check ~push/contact_made_n "
                                  "and the force zero.", adv, self.approach_stall_eps,
                                  self.approach_stall_window,
                                  0.0 if load is None else load, self.contact_made_n)
                    return None, latch, peak

                if (time.time() - t_start) > self.approach_timeout:
                    rospy.logwarn("APPROACH: no contact within %.1fs (peak %.1fN).",
                                  self.approach_timeout, peak)
                    return "no_contact", latch, peak
                rate.sleep()
        except Exception:
            rospy.logerr("Exception during approach - stopping arm, no retry.",
                         exc_info=True)
            self._stop_arm()
            return "error", latch, peak
        # Only reachable on rospy shutdown; treat it like a preempt.
        return "preempted", latch, peak

    def _pivot_in_root(self, goal, root):
        """
        (x, y, z) of the object's yaw pivot in `root`: the centroid of its support
        polygon, at floor height. None when the test cannot run (no collision cloud,
        no object TF), which every caller reads as "fall back to the push axis".

        root is odom or vision, both gravity-aligned, so the lowest-points rule that
        defines the support polygon means what it says.
        """
        if self.collision_pts is None:
            return None
        pose = self._object_pose_in_root(self._object_frame(goal), root=root)
        if pose is None:
            return None
        R, t = pose
        cloud = self.collision_pts @ R.T + t
        xy = el.support_pivot_xy(cloud, self.aim_support_eps)
        if xy is None:
            return None
        return np.array([xy[0], xy[1], float(cloud[:, 2].min())])

    def _align_aim(self, goal, contact_world, push_dir, root):
        """
        (aim_xy, pivot, ang_off) for the base pre-walk. aim_xy is the unit heading
        the body should face, pivot is the yaw pivot (or None), ang_off is the signed
        departure from the push axis.

        aim_xy is always usable: with ~align/aim_mode = push_dir, or with no pivot to
        aim at, it IS the push direction and ang_off is zero.
        """
        d = np.asarray(push_dir, dtype=np.float64)[:2]
        if self.aim_mode != "pivot":
            return d / max(np.linalg.norm(d), 1e-9), None, 0.0
        pivot = self._pivot_in_root(goal, root)
        aim, ang, clamped, why = el.aim_dir_to_pivot(
            contact_world[:2], None if pivot is None else pivot[:2], push_dir,
            self.aim_cone, min_lever=self.aim_min_lever)
        if aim is None:
            return None, pivot, 0.0
        if why:
            rospy.logwarn("ALIGN: aiming down the push axis instead of the yaw "
                          "pivot: %s. An off-axis contact will yaw the object away "
                          "from the crawl.", why)
        elif clamped:
            rospy.logwarn("ALIGN: pivot aim clamped to the %.0fdeg cone. The crawl "
                          "keeps some yaw moment on the object.",
                          np.rad2deg(self.aim_cone))
        else:
            self._vlog("ALIGN[base]: aiming at the yaw pivot (%.3f, %.3f), %+.1fdeg "
                       "off the push axis.", pivot[0], pivot[1], np.rad2deg(ang))
        return aim, pivot, ang

    def _walk_base_align(self, goal, contact_world, push_dir, root):
        # Position the body base_standoff+approach_eps behind the contact, facing the
        # object's yaw pivot (see _align_aim; the push axis under aim_mode=push_dir).
        # Returns True on arrival, False on failure.
        aim, pivot, ang = self._align_aim(goal, contact_world, push_dir, root)
        goal_xy = None if aim is None else el.base_align_pose(
            contact_world, [aim[0], aim[1], 0.0],
            self.base_standoff + self.align_standoff)
        if goal_xy is None:
            # near-vertical push: no meaningful heading, leaving the base where it is.
            rospy.logwarn("ALIGN: push near-vertical, skipping base pre-walk.")
            return True
        bx, by, yaw = goal_xy
        self._publish_aim_viz(root, contact_world, aim, pivot, push_dir)
        self._vlog("ALIGN[base]: target x=%.3f y=%.3f yaw=%.3frad in %s.",
                   bx, by, yaw, root)
        try:
            # Cap base speed so the alignment walk is not abrupt.
            speed = geometry_pb2.SE2VelocityLimit(
                max_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=self.base_max_lin_vel,
                                             y=self.base_max_lin_vel),
                    angular=self.base_max_ang_vel))
            params = spot_command_pb2.MobilityParams(vel_limit=speed)
            cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                bx, by, yaw, frame_name=root, params=params)
            cid = self.command_client.robot_command(
                cmd, end_time_secs=time.time() + self.base_walk_timeout)
            return block_for_trajectory_cmd(self.command_client, cid,
                                            timeout_sec=self.base_walk_timeout)
        except Exception:
            rospy.logerr("ALIGN base pre-walk failed.", exc_info=True)
            return False

    def _push_loop(self, goal, root, latch, approach_peak, approach_reason=None):
        """
        Push phase: continuously issue the hybrid force command while tracking the object.
        Entered already in contact, with the latch armed by _approach_loop.

        The base is PLANTED for the whole phase, so this is a pure arm motion and the
        arm's own travel (_reach_gauge) is the honest limit. The one exception is the
        body assist: if the arm runs out of travel with the tip still loaded, _body_assist
        takes over and finishes the push with the body - in "topple" mode when the object
        is part-way through tipping over, otherwise in "reach" mode to carry the slide out
        to max_finger_reach.

        approach_reason, when set, is the terminal reason the approach failed; the push
        never runs and this only reports the result. approach_peak is the approach's
        peak EE force, logged but deliberately NOT folded into PushResult.peak_force.
        """
        # Push-point start = world contact at the moment the force drive begins.
        start = self._track(goal)
        start_world = start[0] if start is not None else None
        # Elevation of the push-start force direction. The assist bounds its
        # commanded orientation against this, so "how far has the pitch wandered
        # from what the policy asked for" is measured from the push, not from
        # whatever pose the arm was in when it ran out of travel.
        start_elev = None
        if start is not None:
            start_elev = el.quat_elevation(el.task_frame_from_force(start[1])[1])
        object_frame = self._object_frame(goal)
        # Frame the tilt gauge measures in, latched for the whole push: tilt_ref and
        # every later sample have to live in one frame or the delta is meaningless, so
        # a mid-push fallback to the push root is not allowed. Fall back only here, at
        # the start, and say so - the fallback silently reintroduces the TF lag that
        # ~topple/tilt_root_frame exists to remove.
        tilt_root = self.tilt_root_frame or root
        # Pose reference for the topple detector: tilt is the yaw-free rotation picked
        # up since the push began, so however the object happened to be standing at
        # that instant reads as zero tilt.
        tilt_ref = self._tilt_rotation(object_frame, tilt_root)
        if tilt_ref is None and tilt_root != root:
            rospy.logwarn("tilt root %s unavailable at push start; falling back to %s. "
                          "Tilt will lag the object by the driver's TF latency.",
                          tilt_root, root)
            tilt_root = root
            tilt_ref = self._tilt_rotation(object_frame, tilt_root)
        # Push peak only. The approach's peak is reported separately: with a force-mode
        # approach the arrival impact routinely dwarfed the push (24-66 N in bag
        # spot_push_policy_6), so folding the two together made peak_force describe the
        # collision rather than the push.
        peak = 0.0
        ceiling = self.force_clip + self.watchdog_margin
        rate = rospy.Rate(self.loop_rate)
        # The macro-step clock covers the ramp plus goal.duration, the sim's push budget
        # (max_push_steps / sim fps). The sim puts full force on the object at its very
        # first substep; a real push ramps in, so charging the ramp to that budget would
        # leave the object under full force for a fraction of it (ramp_time alone
        # outlasts the whole budget). Armed here because contact is already made.
        deadline = None
        if latch.made_time is not None:
            deadline = latch.made_time + self.ramp_time + float(goal.duration)
        goal_deadline_init = deadline
        end_reason, drift = "timeout", 0.0
        last_force_world = None
        push_dir = None
        # An arm-only command does not cancel the ALIGN pre-walk goal, so the stand has
        # to ride along with every arm command. A stand carries no step plan for the
        # reissue to supersede, so sending it at loop_rate is harmless (same pattern as
        # the retreat RELEASE loop).
        stand_mob = el.build_stand_mobility_command()
        tip_hist = []   # rolling (t, tip projection on push axis) for the stall gauge
        tilt_hist = []  # rolling (t, tilt rad) for the topple duration override
        waived = 0.0    # seconds the duration override has bought this push
        # Ceilings for the override, fixed when the clock arms so repeated extensions
        # cannot walk the deadline forward indefinitely.
        goal_deadline = goal_deadline_init
        hard_cap = (None if goal_deadline_init is None
                    else goal_deadline_init + self.topple_max_beyond_duration)
        # last assist verdict line; replayed un-throttled at exit so every push leaves
        # exactly one log entry explaining why an assist did or did not run.
        verdict = None
        # Held across ticks, not recomputed from zero: a dropped object TF is not
        # evidence that the object stood back up, and zeroing it there would drop the
        # topple predicate for exactly as long as the dropout lasts.
        tilt_gain = 0.0

        if approach_reason is not None:
            end_reason = approach_reason
        else:
            # Phase enum + log only, deliberately not _phase_mark: its phase_pause
            # would idle here, and the handover from the position-held contact to the
            # force ramp has to be immediate.
            self.phase_pub.publish(UInt8(ExecutePushFeedback.PHASE_PUSH))
            rospy.loginfo("PHASE -> PUSH (contact made, approach peak %.1fN); macro "
                          "clock armed for %.2fs ramp + %.2fs push",
                          approach_peak, self.ramp_time, goal.duration)

        try:
            # A failed approach skips the loop entirely and falls through to the
            # result, so there is exactly one place that ends a goal.
            while approach_reason is None and not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    end_reason = "preempted"
                    break

                # One state fetch per tick: the force reading, the reach gauge and the
                # topple latch pose all have to describe the same instant.
                st = self.state_client.get_robot_state()
                snap = st.kinematic_state.transforms_snapshot

                R_tilt = self._tilt_rotation(object_frame, tilt_root)
                if R_tilt is not None:
                    if tilt_ref is None:
                        tilt_ref = R_tilt
                    tilt_gain = el.tilt_since(tilt_ref, R_tilt)
                    self._sample_series(tilt_hist, tilt_gain)

                track = self._track(goal)
                if track is not None:
                    contact_world, force_world = track
                    last_force_world = force_world
                    if start_world is None:
                        start_world = contact_world
                    drift = float(np.linalg.norm(contact_world - start_world))
                    # Force ramp, instead of suddenly applying full force. It starts
                    # at push_force_start (0 by default) because the approach is a
                    # position move that commands no force at all.
                    full_mag = float(np.linalg.norm(force_world))
                    seed = min(self.push_force_start, full_mag)
                    if self.ramp_time > 0.0 and latch.made_time is not None:
                        frac = min(1.0,
                                   (time.time() - latch.made_time) / self.ramp_time)
                        desired_mag = seed + frac * (full_mag - seed)
                    else:
                        desired_mag = full_mag
                    force_cmd = force_world * (desired_mag / max(full_mag, 1e-9))
                    push_dir = force_world / max(full_mag, 1e-9)
                    # ARM: force command in odom (mode 1 — force must point in world),
                    # reissued every tick. The latest command supersedes the running
                    # trajectory at once so force is continuous; cmd_horizon > loop
                    # period -> never expires. The stand rides along to hold the base
                    # still: without it the ALIGN pre-walk goal would keep running,
                    # since an arm-only command cancels no mobility sub-command.
                    cmd, _, _ = el.build_arm_cartesian_command(
                        contact_world, force_cmd, root_frame=root,
                        duration_s=self.cmd_horizon, force_clip=self.force_clip,
                        tool_offset=self._tool_offset, mobility_command=stand_mob)
                    self.command_client.robot_command(cmd)
                    # Stall gauge: with the base planted, a tip that stops advancing
                    # along the push axis while full force is commanded means the arm
                    # is out of travel, whatever the reach heuristic says.
                    root_T_hand = get_a_tform_b(snap, root, HAND_FRAME_NAME)
                    tip_now = np.array(root_T_hand.transform_point(*self._tip_from_hand))
                    self._sample_series(tip_hist, float(np.dot(tip_now, push_dir)))
                    if self.verbose:
                        ramping = (self.ramp_time > 0.0 and latch.made_time is not None
                                   and (time.time() - latch.made_time) < self.ramp_time)
                        rospy.loginfo_throttle(
                            1.0, "PUSH[%s]: |F_cmd|=%.1fN drift=%.3fm tilt=%.1fdeg",
                            "ramp" if ramping else "full", desired_mag, drift,
                            np.rad2deg(tilt_gain))

                load, m = self._contact_force(st)
                cur = 0.0 if m is None else m
                self.force_pub.publish(Float32(cur))
                # Contact decisions run on the bias-removed load; the watchdog and
                # peak below run on the raw estimate, which is the real arm load.
                latch.update(load, time.time())
                if m is not None:
                    peak = max(peak, m)
                    if m > ceiling:
                        rospy.logerr("WATCHDOG: EE force %.1fN > %.1fN - stopping arm.",
                                     m, ceiling)
                        self._stop_arm()
                        end_reason = "watchdog"
                        break

                self._publish_feedback(ExecutePushFeedback.PHASE_PUSH, latch.made,
                                       cur, drift)

                # Out of arm travel: the reach heuristic fires, the object has outrun
                # the SMDP radius, or the tip has stopped advancing under load.
                arm_out, tip_reach, reach_lim = self._reach_gauge(st, snap)
                stalled, tip_adv = self._tip_stalled(tip_hist)
                # A tip that is not advancing while the force is still ramping up is an
                # unloaded object, not an arm out of travel: early in the ramp the
                # command is a few N against a goal of tens. Only trust the gauge once
                # full commanded force has been on the object for a whole window.
                if (latch.made_time is None
                        or (time.time() - latch.made_time)
                        < (self.ramp_time + self.topple_stall_window)):
                    stalled, tip_adv = False, float("nan")
                drift_out = drift > self.max_finger_reach
                out_of_travel = arm_out or drift_out or stalled
                # Mid-topple: the object has rotated away from its push-start pose AND
                # the tip is still loaded, i.e. we are the ones tipping it. Force alone
                # cannot tell a topple from a wedged slide, tilt alone cannot tell it
                # from a bump.
                # Debounced, so a one-tick dropout cannot block the assist at the very
                # moment the object starts tipping.
                loaded = latch.loaded
                toppling = tilt_gain > self.topple_tilt_min and loaded
                # Which mode, if any, the body assist would run in right now. Only the
                # topple case waives the SMDP radius, so reach mode is gated on the push
                # still owing drift; toppling wins when both would apply.
                mode = None
                if latch.made and out_of_travel and push_dir is not None:
                    if toppling:
                        mode = "topple"
                    elif self.reach_assist and not drift_out:
                        mode = "reach"
                # Why no assist fired, published and logged so a mistuned threshold is
                # visible instead of surfacing only as a quiet "arm_limit".
                verdict = self._assist_report(
                    mode=mode, tilt_gain=tilt_gain, force=m, contact_made=latch.made,
                    arm_out=arm_out, tip_reach=tip_reach, reach_lim=reach_lim,
                    drift=drift, drift_out=drift_out, stalled=stalled,
                    tip_adv=tip_adv, out_of_travel=out_of_travel, toppling=toppling,
                    loaded=loaded, low_for=latch.low_for, waived=waived)
                if mode is not None:
                    # Waive both reach limits and finish the push with the body.
                    end_reason, peak, drift = self._body_assist(
                        goal, root, snap, push_dir, start_world, peak, drift, ceiling,
                        mode=mode, deadline=deadline, start_elev=start_elev)
                    break
                if arm_out:
                    # Arm ran out of travel with the base planted. Distinct from
                    # "reach", which is the object outrunning max_finger_reach.
                    end_reason = "arm_limit"
                    break

                # SMDP stops when push point is out of reach or contact is lost.
                if drift > self.max_finger_reach:
                    end_reason = "reach"
                    break
                if latch.lost:
                    end_reason = "contact_lost"
                    break
                # deadline is None until contact latches, so the approach is bounded
                # by the reach gauge and the watchdog, not by the macro-step clock.
                if deadline is not None and time.time() >= deadline:
                    # Mid-topple with the tilt still climbing: buy one more tick instead
                    # of dropping the object mid-tip. Every other exit (contact lost,
                    # reach, watchdog, preempt) is checked above and still ends the push.
                    rise, rise_valid = el.tilt_rise(tilt_hist,
                                                    self.topple_stall_window)
                    if (self.topple_duration_override and toppling and rise_valid
                            and rise > self.topple_tilt_rise_eps
                            and time.time() < hard_cap):
                        waived = time.time() - goal_deadline
                        deadline = time.time() + 1.0 / self.loop_rate
                        rospy.loginfo_throttle(
                            1.0, "PUSH: duration override, mid-topple tilt=%.1fdeg "
                            "(+%.2fdeg/%.2fs) waived=%.1f/%.1fs",
                            np.rad2deg(tilt_gain), np.rad2deg(rise),
                            self.topple_stall_window, waived,
                            self.topple_max_beyond_duration)
                    else:
                        end_reason = "timeout"
                        break
                rate.sleep()
        except Exception:
            # for safety, stop the arm, do not retry. Bring the node down safely.
            rospy.logerr("Exception during push - stopping arm, no retry.", exc_info=True)
            self._stop_arm()
            self._abort("error", "push exception", peak=peak,
                        contact_made=latch.made, drift=drift)
            rospy.signal_shutdown("push exception")
            return

        # Marker at the final tracked pose corresponding to the best-effort.
        if last_force_world is not None:
            track = self._track(goal)
            if track is not None:
                self._publish_marker(root, track[0], last_force_world,
                                     float(min(np.linalg.norm(last_force_world),
                                               self.force_clip)))

        # peak_force is the PUSH peak. The approach peak is logged beside it rather
        # than folded in: they measure different things, and mixing them is what made
        # the reported peak an arrival impact instead of a push force.
        res = ExecutePushResult(contact_made=latch.made, peak_force=peak,
                                finger_travel=drift, end_reason=end_reason)
        rospy.loginfo("PushResult: contact=%s peak=%.1fN (approach peak %.1fN) "
                      "drift=%.3fm reason=%s%s",
                      latch.made, peak, approach_peak, drift, end_reason,
                      "" if waived <= 0.0
                      else " (duration override +%.1fs)" % waived)
        # Un-throttled, so a push that ended in one tick still explains itself.
        rospy.loginfo("ASSIST verdict: %s",
                      verdict if verdict is not None else "no push tick completed")
        # Home the arm before returning the result.
        self._return_to_ready(goal, root, last_force_world)
        if end_reason == "preempted":
            self.server.set_preempted(res)
        else:
            self.server.set_succeeded(res)

    def _sample_series(self, hist, value, window=None):
        # Append (now, value) and drop samples older than the stall window, so hist[0]
        # is always the oldest sample still in window. Shared by the tip-stall gauge,
        # the topple duration override and the approach block gauge; `window` picks
        # which of those windows this series is kept to.
        window = self.topple_stall_window if window is None else window
        now = time.time()
        hist.append((now, value))
        cutoff = now - window
        while len(hist) > 2 and hist[1][0] < cutoff:
            hist.pop(0)

    def _tip_stalled(self, hist, window=None, eps=None):
        # (stalled, advance): stalled when the tip has advanced less than eps along
        # the push axis over a full window. Needs a window's worth of samples first,
        # otherwise the gauge fires on the first tick of every push. The advance is
        # returned so the assist report can print the raw number.
        window = self.topple_stall_window if window is None else window
        eps = self.topple_stall_eps if eps is None else eps
        if len(hist) < 2 or (hist[-1][0] - hist[0][0]) < window:
            return False, float("nan")
        adv = hist[-1][1] - hist[0][1]
        return adv < eps, adv

    def _assist_report(self, mode, tilt_gain, force, contact_made, arm_out, tip_reach,
                       reach_lim, drift, drift_out, stalled, tip_adv, out_of_travel,
                       toppling, loaded, low_for, waived):
        """
        Format, publish and return one line saying whether a body assist fired and, if
        not, which term of the predicate blocked it.

        The assist predicate ANDs five terms, so a mistuned threshold shows up only as a
        push that quietly ends in "arm_limit". Every gauge prints as value/threshold so
        the line is readable without the source, and `block=` names the failing terms.
        """
        tilt_ok = tilt_gain > self.topple_tilt_min
        # Terms are reported in the order the gate evaluates them. `block` is empty when
        # an assist fired, and otherwise lists every term that would have to change for
        # either mode to become eligible.
        block = []
        if mode is not None:
            pass
        elif not contact_made:
            block.append("no_contact")
        else:
            if not out_of_travel:
                block.append("travel")
            if not tilt_ok:
                block.append("tilt")
            if not loaded:
                block.append("load")
            if not self.reach_assist:
                block.append("reach_disabled")
            elif drift_out:
                block.append("drift")
        line = ("tilt=%.2f/%.2fdeg %s | F=%s/%.1fN %s low=%.2f/%.2fs | "
                "arm_out=%d %s=%.3f/%.3f | "
                "drift=%.3f/%.3fm %d | stall=%d adv=%s/%.3fm || contact=%d travel=%d "
                "topple=%d waived=%.1f/%.1fs assist=%s block=%s"
                % (np.rad2deg(tilt_gain), self.topple_tilt_min_deg,
                   "PASS" if tilt_ok else "FAIL",
                   "n/a" if force is None else "%.1f" % force, self.contact_made_n,
                   "PASS" if loaded else "FAIL",
                   low_for, self.contact_lost_grace,
                   arm_out, self.reach_signal, tip_reach, reach_lim,
                   drift, self.max_finger_reach, drift_out,
                   stalled,
                   "n/a" if np.isnan(tip_adv) else "%+.3f" % tip_adv,
                   self.topple_stall_eps,
                   contact_made, out_of_travel, toppling,
                   waived, self.topple_max_beyond_duration,
                   mode or "none", ",".join(block) if block else "none"))
        self.assist_pub.publish(String(line))
        if self.verbose:
            rospy.loginfo_throttle(1.0, "ASSIST: %s", line)
        return line

    def _body_assist(self, goal, root, snap, push_dir, start_world, peak, drift,
                     ceiling, mode, deadline, start_elev=None):
        """
        Finish a push with the BODY once the arm has run out of travel.

        Three stages, the first two topple-only. RECONTACT (see _await_recontact)
        stands still and waits out the object's break-away, latching the pose to hold
        at the instant the object falls back onto the tool - without it every later
        stage latches an arm that is still travelling in force mode and then commands
        it backwards. RECENTER (see _recenter) holds that tip fixed in `root` while
        the base crawls, so the arm folds back out of its far-reach singularity and
        gets its position AND orientation authority back. Then the arm is frozen
        rigid relative to the body (all-position Cartesian command rooted in
        flat_body, see el.build_body_locked_arm_command) and the base keeps crawling,
        so leg drive reaches the object through the arm as a strut.

        The commanded tip POSITION is latched here and never recomputed: re-deriving
        it per tick from the live object would let the object pull the hand along
        instead of the hand pushing the object. The commanded ORIENTATION is not
        frozen with it when ~topple/recenter_track_orientation is set - it keeps
        following the tracked push force, which rotates with the tipping object.
        Orientation cannot drag the tip anywhere, and holding the push-start pitch
        through a topple is precisely wrong: the face being pushed is rotating away.

        The arm reach limits are waived for the duration in both modes; the exits
        differ:

        mode="topple" - the object is part-way through tipping over. The SMDP radius and
            goal.duration are BOTH ignored: a topple routinely outlasts push_dwell, and
            cutting it mid-tip is the failure this whole path exists to prevent. It ends
            when the contact force collapses (the object went past its tipping point),
            when the tip never re-loads, at the body-travel cap, or when the body box
            runs out of clearance to the object.

        mode="reach" - the object is NOT tipping; the arm simply ran out of travel before
            the push reached max_finger_reach. This is an ordinary push continuation, so
            it honours both the SMDP radius (-> "reach") and the macro-step budget
            (-> "timeout"), and its travel cap is additionally floored by the drift the
            push still owes so the body never walks further than the push needs. It does
            NOT recenter: recentering spends body travel without advancing the push, and
            a slide has no tipping face whose pitch has to be held.

        NOTE: mode is fixed for the lifetime of the assist. An object that only starts
        tipping after the arm ran out of travel therefore finishes the push in reach
        mode. See npm_control/doc/reach_to_topple_promotion.md.

        start_elev is the elevation (rad) of the push-start force direction, the
        reference the commanded orientation is bounded against. None -> use the
        measured wrist elevation at the handover.

        Returns (end_reason, peak, drift).
        """
        # Both modes are the same physical mode - arm locked to the body, base driving -
        # so they share PHASE_TOPPLE on the phase/feedback topics. end_reason and the
        # assist_debug line distinguish them, including the recenter stage.
        self._phase_mark("TOPPLE")
        # Contact state spans the recenter and crawl stages. The handover from force
        # mode to a position hold drops the contact force for a moment, so the
        # collapse exit only arms once the tip has genuinely re-loaded against the
        # object (latch.made), and then only after the force has stayed collapsed for
        # contact_lost_grace. Without both the assist quits on its own transient.
        # RECONTACT keeps its own latches and replaces this one on the way out: the
        # break-away it waits through is expected, and load_grace should run from the
        # re-contact rather than from the handover.
        latch = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                self.contact_lost_grace)
        entered = time.time()
        if start_elev is None:
            root_T_wr1 = get_a_tform_b(snap, root, WR1_FRAME_NAME)
            qm = root_T_wr1.rot
            start_elev = el.quat_elevation((qm.w, qm.x, qm.y, qm.z))
        # The crawl is body +x in both stages, and it only pushes the object without
        # spinning it while the body faces the object's yaw PIVOT. ALIGN aimed it
        # there, but the push has been yawing the object ever since, so the aim is
        # stale by now. Both stages steer it back (see _aim_yaw_rate); say so loudly
        # if the handover starts far off.
        root_T_body = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME)
        aim_err, _pivot, _tip = self._aim_heading_err(goal, snap, root)
        aim_src = "pivot"
        if aim_err is None:
            aim_src = "push axis"
            yaw = root_T_body.rot.to_yaw()
            aim_err = el.wrap_pi(float(np.arctan2(push_dir[1], push_dir[0])) - yaw)
        if abs(aim_err) > np.deg2rad(30.0):
            rospy.logwarn("ASSIST[%s]: body heading is %+.0fdeg off the %s; the "
                          "forward crawl will not track it.%s",
                          mode, np.rad2deg(aim_err), aim_src,
                          "" if self.aim_track else " Aim tracking is OFF.")

        q_cmd_root = None
        tip_root = None
        if mode == "topple" and self.recontact:
            # Stand still until the object is back on the tool, and take the pose to
            # hold from THAT instant. Everything below - the recenter's world-pinned
            # tip and the crawl's body-locked one - is latched off what this returns.
            abort, peak, drift, snap, tip_root, q_cmd_root, _why = \
                self._await_recontact(goal, root, peak, drift, ceiling, start_world,
                                      start_elev)
            if abort is not None:
                self._plant_base(mode)
                rospy.loginfo("ASSIST[%s] done: reason=%s (during recontact) "
                              "drift=%.3fm peak=%.1fN", mode, abort, drift, peak)
                return abort, peak, drift
            # The break-away above is expected, not a collapse, so it must not arm
            # the exits the crawl runs on. Restart the shared latch and its grace
            # clock from the re-contact, which is where the assist really begins.
            latch = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                    self.contact_lost_grace)
            # ...and start it ARMED. _await_recontact returns without an abort only
            # when contact is confirmed present - "held" is recontact_settle of
            # continuous load, "recontact" is the return latching - so this states a
            # measured fact rather than assuming one. Re-arming from zero would
            # instead demand contact_made_n (5 N) from an object that has just
            # settled onto a stationary tool: RECONTACT counts that object as loaded
            # at contact_eps (3 N), and a settled object sits in the 3-5 N hysteresis
            # band, so the re-load never comes and recenter quits on its own
            # handover. Bag spot_push_policy_10 pushes 1 and 2 died exactly there
            # (raw 4.0-4.3 N and 1.5-2.5 N, abort "contact_lost" 0.5-0.6 s in);
            # push 3 survived only because its load happened to drift over 5 N.
            # A real collapse still ends the assist: below contact_eps for
            # contact_lost_grace sets latch.lost, which reads as "toppled".
            latch.force_made(time.time())
            entered = time.time()

        if mode == "topple" and self.recenter:
            abort, peak, drift, snap, q_cmd_root = self._recenter(
                goal, root, snap, peak, drift, ceiling, latch, entered, start_world,
                start_elev, tip_root=tip_root, q_cmd=q_cmd_root)
            if abort is not None:
                # Recenter never handed the crawl a loaded, workable arm. Plant the
                # base here rather than falling through: crawling with the arm in
                # whatever pose recenter left it is the failure mode this stage
                # exists to prevent.
                self._plant_base(mode)
                rospy.loginfo("ASSIST[%s] done: reason=%s (during recenter) "
                              "drift=%.3fm peak=%.1fN", mode, abort, drift, peak)
                return abort, peak, drift

        # Latch the tool pose in flat_body, from the POST-recenter configuration.
        # wrist_tform_tool is a pure translation, so the tool and the wrist share an
        # orientation.
        flat_T_wr1 = get_a_tform_b(snap, GRAV_ALIGNED_BODY_FRAME_NAME, WR1_FRAME_NAME)
        tip_body = flat_T_wr1.transform_point(self.tool_tip_x, 0.0, self.tool_tip_z)
        q_body = flat_T_wr1.rot
        quat_wxyz = (q_body.w, q_body.x, q_body.y, q_body.z)
        if q_cmd_root is not None:
            # Freeze what the previous stage COMMANDED, not what the arm settled at.
            # Under load at extension the measured wrist carries the tracking error
            # the recenter exists to shed, and latching the measurement bakes it into
            # the strut.
            quat_wxyz = self._root_quat_to_flat(snap, root, q_cmd_root)
        # Travel origin: body position at the latch, in the (fixed) root frame. Taken
        # after recenter, so the crawl gets its full budget and the recenter travel is
        # bounded separately by recenter_max_travel_cap.
        root_T_body = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME)
        origin_xy = np.array([root_T_body.x, root_T_body.y])
        # Body yaw at the latch. The tip is latched in flat_body, so every radian the
        # base yaws afterwards would sweep the tip sideways across the pushed face;
        # el.counter_yaw_tip rotates the commanded offset back by exactly this delta,
        # leaving only the body's TRANSLATION to drive the tip into the object.
        yaw0 = root_T_body.rot.to_yaw()

        # Travel cap. In reach mode the body only has to make up the drift the push
        # still owes, so cap it there (plus a small margin for the object lagging the
        # tip) rather than letting it walk the full topple budget.
        if mode == "reach":
            cap = min(self.reach_max_body_travel,
                      max(0.0, self.max_finger_reach - drift) + 0.1)
        else:
            cap = self.topple_max_body_travel
        self._vlog("ASSIST[%s]: locking tip at body (%.3f, %.3f, %.3f) elev=%.1fdeg, "
                   "crawling forward at %.2fm/s (cap %.2fm, drift %.3f/%.3fm).",
                   mode, tip_body[0], tip_body[1], tip_body[2],
                   np.rad2deg(el.quat_elevation(quat_wxyz)), self.topple_body_vel,
                   cap, drift, self.max_finger_reach)

        rate = rospy.Rate(self.loop_rate)
        end_reason, travel = "topple_travel" if mode == "topple" else "assist_travel", 0.0
        last_tick = time.time()
        try:
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    end_reason = "preempted"
                    break

                st = self.state_client.get_robot_state()
                snap = st.kinematic_state.transforms_snapshot
                load, m = self._contact_force(st)
                cur = 0.0 if m is None else m
                self.force_pub.publish(Float32(cur))
                latch.update(load, time.time())
                if m is not None:
                    peak = max(peak, m)
                    if m > ceiling:
                        # The assist is position-mode, so the contact force is
                        # geometric rather than commanded; this watchdog is the only
                        # ceiling on it.
                        rospy.logerr("WATCHDOG: EE force %.1fN > %.1fN - stopping arm.",
                                     m, ceiling)
                        self._stop_arm()
                        end_reason = "watchdog"
                        break

                root_T_body = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME)
                travel = float(np.linalg.norm(
                    np.array([root_T_body.x, root_T_body.y]) - origin_xy))
                # Live geometry, not a fixed number: the object is tipping, so how
                # much room the body has in front of it changes every tick.
                clearance, hits = self._clearance(goal)
                self._publish_clearance_viz(clearance, hits)

                aim_err, pivot, tip_live = self._aim_heading_err(goal, snap, root)
                self._publish_aim_viz(root, tip_live, None, pivot,
                                      self._body_x(root_T_body), throttle=True)
                d_yaw = el.wrap_pi(root_T_body.rot.to_yaw() - yaw0)
                v_rot = self._aim_yaw_rate(aim_err, d_yaw)
                tip_cmd = el.counter_yaw_tip(tip_body, d_yaw)

                now = time.time()
                dt, last_tick = now - last_tick, now
                quat_cmd = quat_wxyz
                if self.recenter_track_orientation:
                    root_T_wr1 = get_a_tform_b(snap, root, WR1_FRAME_NAME)
                    qm = root_T_wr1.rot
                    q_cmd_root = self._track_quat(goal, (qm.w, qm.x, qm.y, qm.z),
                                                  q_cmd_root, start_elev, dt)
                    quat_cmd = self._root_quat_to_flat(snap, root, q_cmd_root)

                # Base velocity + body-locked arm, in one synchronized command. A
                # velocity command carries no step plan for the reissue to supersede,
                # so unlike the SE2 goals elsewhere this is safe at loop_rate.
                # end_time_secs is REQUIRED: it is the only thing that fills
                # se2_velocity_request.end_time.
                mob = el.build_velocity_mobility_command(
                    self.topple_body_vel, v_rot=v_rot,
                    max_lin_vel=self.topple_body_vel,
                    max_ang_vel=max(self.aim_yaw_rate, 1e-3))
                cmd = el.build_body_locked_arm_command(
                    tip_cmd, quat_cmd, duration_s=self.cmd_horizon,
                    tool_offset=self._tool_offset, mobility_command=mob)
                self.command_client.robot_command(
                    cmd, end_time_secs=time.time() + self.topple_vel_end_time)

                track = self._track(goal)
                if track is not None and start_world is not None:
                    # Keep reporting drift against the ORIGINAL push start, so
                    # finger_travel stays comparable across the whole macro-step.
                    drift = float(np.linalg.norm(track[0] - start_world))
                self._publish_feedback(ExecutePushFeedback.PHASE_TOPPLE, latch.loaded,
                                       cur, drift)
                line = ("stage=crawl assist=%s travel=%.3f/%.3fm drift=%.3f/%.3fm "
                        "F=%.1fN loaded=%d low=%.2f/%.2fs elev=%+.1fdeg clr=%s "
                        "aim_err=%s yaw=%+.0f/%.0fdeg v_rot=%+.2frad/s"
                        % (mode, travel, cap, drift, self.max_finger_reach, cur,
                           latch.loaded, latch.low_for, self.contact_lost_grace,
                           np.rad2deg(el.quat_elevation(quat_cmd)),
                           self._clearance_str(clearance),
                           self._aim_err_str(aim_err), np.rad2deg(d_yaw),
                           np.rad2deg(self.aim_yaw_max), v_rot))
                self.assist_pub.publish(String(line))
                if self.verbose:
                    rospy.loginfo_throttle(0.5, "ASSIST: %s%s", line,
                                           "" if latch.made else " [waiting for re-load]")

                if latch.lost:
                    # Force collapsed. In topple mode that means the object went past
                    # its tipping point (success); in reach mode the object is only
                    # sliding, so a collapse can only be a lost contact.
                    end_reason = "toppled" if mode == "topple" else "contact_lost"
                    break
                if not latch.made and (time.time() - entered) > self.topple_load_grace:
                    # The position hold never took the object's weight back up, so
                    # walking further would just drive the body at empty air.
                    end_reason = "contact_lost"
                    break
                if (self.recenter_clearance_gate and clearance is not None
                        and clearance <= 0.0):
                    # The body box has reached the object. Any further crawl is the
                    # chassis pushing, not the arm, and nothing above bounds it.
                    end_reason = "clearance"
                    break
                if travel > cap:
                    end_reason = "topple_travel" if mode == "topple" else "assist_travel"
                    break
                if mode == "reach":
                    # Reach mode is an ordinary push continuation, so unlike a topple it
                    # honours the SMDP stop radius and the macro-step budget.
                    if drift > self.max_finger_reach:
                        end_reason = "reach"
                        break
                    if time.time() >= deadline:
                        end_reason = "timeout"
                        break
                rate.sleep()
        except Exception:
            rospy.logerr("Exception during %s assist - stopping arm, no retry.", mode,
                         exc_info=True)
            self._stop_arm()
            end_reason = "error"
        finally:
            self._plant_base(mode)
        rospy.loginfo("ASSIST[%s] done: reason=%s travel=%.3fm drift=%.3fm peak=%.1fN",
                      mode, end_reason, travel, drift, peak)
        return end_reason, peak, drift

    def _plant_base(self, mode):
        # Plant the base before anything else runs: the velocity command outlives the
        # loop that sent it by up to topple_vel_end_time, and whatever comes next
        # (the retreat, or a returned goal) must not start with the body still
        # crawling into the object.
        try:
            self.command_client.robot_command(
                RobotCommandBuilder.synchro_stand_command())
        except Exception:
            rospy.logwarn("ASSIST[%s]: stand failed; base may still be moving.",
                          mode, exc_info=True)

    def _hold_pose(self, goal, root, snap, start_elev, prev_cmd=None, dt=0.0):
        """
        The pose an assist stage should hold, read out of one transforms snapshot:
        the MEASURED tip in `root` plus the tracked tool orientation.

        The tip is measured, not commanded - it is where contact actually is. The
        commanded contact point rides a moving object, and using it here would put a
        step into the handover.
        """
        root_T_wr1 = get_a_tform_b(snap, root, WR1_FRAME_NAME)
        tip = np.array(root_T_wr1.transform_point(
            self.tool_tip_x, 0.0, self.tool_tip_z))
        qm = root_T_wr1.rot
        q = self._track_quat(goal, (qm.w, qm.x, qm.y, qm.z), prev_cmd, start_elev, dt)
        return tip, q

    def _await_recontact(self, goal, root, peak, drift, ceiling, start_world,
                         start_elev):
        """
        Stand still at the handover, and latch the assist's hold pose at the instant
        the object comes back down onto the tool.

        Why this exists. The arm-only push runs in FORCE mode, so when the reach
        gauge trips the tip is still travelling - 0.5 to 1.7 m/s across bags
        spot_push_policy_8 and _9. Every stage downstream used to latch its hold pose
        out of the `snap` taken at the TOP of that push tick, 40-160 ms old by the
        time the hold command went out. The arm coasts past that pose and the
        position hold then pulls it BACK: 0.07-0.57 m of backward tip travel on those
        two bags, against 0.000-0.028 m on spot_push_policy_6, which predates the
        recenter stage. Worse, the retract lands inside the window where the object
        has tipped away and is touching nothing, so the object falls back onto an arm
        that has retreated from it - measured 0.115-0.239 m short at the moment
        contact returned.

        Two fixes, both here:

        1. The pose comes from a state read taken NOW, after the arm has stopped, so
           commanding it is a no-op instead of a step backwards.
        2. The pose handed on to RECENTER and the crawl is RE-READ at the moment
           contact returns, not at the moment the push ended.

        The stage costs the assist nothing in object terms: base planted, tip pinned
        in `root`, so the object is neither pushed nor released. Only the orientation
        moves, tracking the tipping face through _track_quat exactly as the recenter
        does.

        Two ways in. Loaded at the handover (the usual case - the bags show 15-75 N
        still on the tool when the gauge trips) means watching for the break-away
        first; already unloaded means going straight to waiting for the return. An
        object that never breaks away at all ends the stage early on
        ~topple/recontact_settle, so a slide-like topple does not pay the budget.

        Contact bookkeeping is LOCAL to this stage. The object leaving the tool here
        is the expected behaviour, not a collapse, so it must never reach the latch
        the crawl exits on; _body_assist starts a fresh one from the re-contact.

        Returns (abort_or_None, peak, drift, snap, tip_root, q_cmd, reason), where
        reason is "recontact" (it came back) or "held" (it never left). The only
        abort is "toppled": the object went over and never returned inside the
        budget, plus the usual watchdog/preempt/error.
        """
        st = self.state_client.get_robot_state()
        snap = st.kinematic_state.transforms_snapshot
        tip_root, q_cmd = self._hold_pose(goal, root, snap, start_elev)
        load, m = self._contact_force(st)

        # `latch` watches the break-away, `back` the return. Two latches because both
        # `made` and `lost` are sticky: one cannot report a loss and then a fresh make.
        latch = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                self.contact_lost_grace)
        started = time.time()
        latch.update(load, started)
        waiting = not latch.made
        back = None
        loaded_since = None if waiting else started
        stand_mob = el.build_stand_mobility_command()
        reach_in = el.tip_reach_from_shoulder(snap, self._tip_from_hand)
        self._vlog("ASSIST[topple]: recontact - holding tip at root (%.3f, %.3f, %.3f) "
                   "elev=%.1fdeg reach=%.3fm, %s (settle %.2fs, budget %.1fs).",
                   tip_root[0], tip_root[1], tip_root[2],
                   np.rad2deg(el.quat_elevation(q_cmd)), reach_in,
                   "unloaded at the handover - waiting for the object to fall back"
                   if waiting else "loaded - watching for the break-away",
                   self.recontact_settle, self.recontact_wait)

        rate = rospy.Rate(self.loop_rate)
        last_tick = started
        abort, reason = None, ("recontact" if waiting else "held")
        clearance, elapsed = None, 0.0
        broke_at = None
        try:
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    abort = "preempted"
                    break

                st = self.state_client.get_robot_state()
                snap = st.kinematic_state.transforms_snapshot
                load, m = self._contact_force(st)
                cur = 0.0 if m is None else m
                self.force_pub.publish(Float32(cur))
                now = time.time()
                elapsed = now - started
                latch.update(load, now)
                if m is not None:
                    peak = max(peak, m)
                    if m > ceiling:
                        rospy.logerr("WATCHDOG: EE force %.1fN > %.1fN - stopping arm.",
                                     m, ceiling)
                        self._stop_arm()
                        abort = "watchdog"
                        break

                dt, last_tick = now - last_tick, now
                # Only the ORIENTATION moves here: the tip stays pinned at the pose
                # latched above, so the arm holds its ground while the face it is
                # pointing at rotates away and back.
                tip_meas, q_cmd = self._hold_pose(goal, root, snap, start_elev,
                                                  q_cmd, dt)
                cmd = el.build_body_locked_arm_command(
                    tip_root, q_cmd, root_frame=root,
                    remain_near_current_joints=False, duration_s=self.cmd_horizon,
                    tool_offset=self._tool_offset, mobility_command=stand_mob)
                self.command_client.robot_command(cmd)

                track = self._track(goal)
                if track is not None and start_world is not None:
                    drift = float(np.linalg.norm(track[0] - start_world))
                clearance, hits = self._clearance(goal)
                self._publish_clearance_viz(clearance, hits)
                self._publish_feedback(ExecutePushFeedback.PHASE_TOPPLE, latch.made,
                                       cur, drift)
                if load is not None and load >= self.contact_eps:
                    if loaded_since is None:
                        loaded_since = now
                else:
                    loaded_since = None

                line = ("stage=recontact assist=topple state=%s "
                        "hold=(%.3f,%.3f,%.3f) reach=%.3fm clr=%s drift=%.3fm "
                        "F=%.1fN loaded=%.2f/%.2fs elev=%+.1fdeg t=%.2f/%.2fs"
                        % ("wait_return" if waiting else "wait_break",
                           tip_root[0], tip_root[1], tip_root[2],
                           el.tip_reach_from_shoulder(snap, self._tip_from_hand),
                           self._clearance_str(clearance), drift, cur,
                           0.0 if loaded_since is None else now - loaded_since,
                           self.recontact_settle,
                           np.rad2deg(el.quat_elevation(q_cmd)),
                           elapsed, self.recontact_wait))
                self.assist_pub.publish(String(line))
                if self.verbose:
                    rospy.loginfo_throttle(0.5, "ASSIST: %s", line)

                if not waiting:
                    if latch.lost:
                        # Expected: the object has tipped off the tool. Keep holding.
                        waiting, broke_at, loaded_since = True, elapsed, None
                        back = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                               self.contact_lost_grace)
                        self._vlog("ASSIST[topple]: recontact - object broke away at "
                                   "t=%.2fs; holding for it to fall back.", elapsed)
                    elif (latch.made and loaded_since is not None
                          and (now - loaded_since) > self.recontact_settle):
                        # It never left. Nothing to wait for; the pose read at entry
                        # is already the loaded one.
                        reason = "held"
                        break
                else:
                    if back is None:
                        back = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                               self.contact_lost_grace)
                    back.update(load, now)
                    if back.made:
                        # THE point of this stage: the object is back on the tool, so
                        # this is the arm pose the assist should hold from here.
                        tip_root, reason = tip_meas, "recontact"
                        break

                if elapsed > self.recontact_wait:
                    if waiting:
                        # It went over and never came back. Crawling now would drive
                        # the body at empty air.
                        abort = "toppled"
                    else:
                        reason = "held"
                    break
                rate.sleep()
        except Exception:
            rospy.logerr("Exception during recontact hold - stopping arm, no retry.",
                         exc_info=True)
            self._stop_arm()
            abort = "error"

        reach_out = el.tip_reach_from_shoulder(snap, self._tip_from_hand)
        rospy.loginfo("ASSIST[topple]: recontact done reason=%s hold=(%.3f, %.3f, "
                      "%.3f) reach=%.3f->%.3fm elev=%+.1fdeg break=%s t=%.2fs clr=%s%s",
                      abort or reason, tip_root[0], tip_root[1], tip_root[2],
                      reach_in, reach_out, np.rad2deg(el.quat_elevation(q_cmd)),
                      "never" if broke_at is None else "%.2fs" % broke_at, elapsed,
                      self._clearance_str(clearance),
                      "" if abort is None else " (ABORT)")
        return abort, peak, drift, snap, tip_root, q_cmd, reason

    def _recenter(self, goal, root, snap, peak, drift, ceiling, latch, entered,
                  start_world, start_elev, tip_root=None, q_cmd=None):
        """
        Base catch-up stage: hold the pusher tip FIXED IN `root` while the base
        crawls forward, so the arm folds back into a part of its envelope where it
        can control tip position and orientation again.

        Why this exists. The arm-only push ends when the arm is out of travel, i.e.
        at full extension, and the old assist latched the strut pose right there.
        An arm at its far-reach limit has no travel left to trade for wrist pitch,
        so it holds position and loses orientation: in bag spot_push_policy_6 the
        commanded tool elevation was tracking the tipping face down from +20.0 to
        +15.8 deg through the push, then climbed to +34.8 deg over the assist while
        the hand rose 13 cm. flat_body is gravity-aligned, so a TRACKED latch would
        have held that elevation flat - the climb is tracking error from an arm with
        nothing left to give.

        Crawling with the tip pinned in world costs nothing physically: the contact
        point does not move, so the object is neither pushed nor released, and the
        arm recovers travel one-for-one with body displacement along the push axis.

        The hold point is fixed in `root`, but the command that carries it is
        all-position in FLAT_BODY, re-derived from that fixed point every tick. Same
        physical target, same fold - and, unlike a `root`-rooted command, a base that
        actually walks. See the measurements at the command site: an odom-rooted arm
        Cartesian command in the same SynchronizedCommand plants the body, so this
        stage used to be a 3 s no-op that only looked like it worked.

        force_remain_near_current_joint_configuration stays OFF - the arm has to
        re-solve continuously as it folds, and that flag damps exactly that.
        Orientation is NOT held: it tracks the push force through _track_quat, so the
        tool keeps pointing into the face that is rotating away from it.

        Ends on the reach gauge (the arm has its envelope back), on the live body
        box clearance to the object, on the travel cap, on the near-body floor, or
        on timeout - all of which continue into the crawl, because a short recenter
        still leaves the arm better off than none. Only a lost contact, a watchdog
        trip or a preempt aborts the assist outright.

        tip_root/q_cmd are the pose to hold, handed over by _await_recontact - read
        at the instant the object came back onto the tool. Deriving them here instead
        is the fallback for ~topple/recontact:false, and it is what produced the
        backward tip step that stage exists to remove: `snap` is the push loop's, one
        tick old, taken while the arm was still travelling in force mode.

        Returns (abort_reason_or_None, peak, drift, snap, commanded_quat_in_root).
        """
        if tip_root is None or q_cmd is None:
            tip_root, q_cmd = self._hold_pose(goal, root, snap, start_elev)
        tip_root = np.asarray(tip_root, dtype=np.float64).reshape(3)
        elev_in = el.quat_elevation(q_cmd)
        root_T_body = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME)
        origin_xy = np.array([root_T_body.x, root_T_body.y])
        reach_in = el.tip_reach_from_shoulder(snap, self._tip_from_hand)
        self._vlog("ASSIST[topple]: recenter - holding tip at root (%.3f, %.3f, %.3f) "
                   "elev=%.1fdeg, crawling at %.2fm/s until reach %.3f -> %.3fm "
                   "(cap %.2fm, timeout %.1fs).",
                   tip_root[0], tip_root[1], tip_root[2], np.rad2deg(elev_in),
                   self.recenter_body_vel, reach_in, self.recenter_target_reach,
                   self.recenter_max_travel_cap, self.recenter_timeout)

        rate = rospy.Rate(self.loop_rate)
        started = time.time()
        last_tick = started
        abort, reason, travel = None, "timeout", 0.0
        reach_now, clearance = reach_in, None
        try:
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    abort = "preempted"
                    break

                st = self.state_client.get_robot_state()
                snap = st.kinematic_state.transforms_snapshot
                load, m = self._contact_force(st)
                cur = 0.0 if m is None else m
                self.force_pub.publish(Float32(cur))
                latch.update(load, time.time())
                if m is not None:
                    peak = max(peak, m)
                    if m > ceiling:
                        rospy.logerr("WATCHDOG: EE force %.1fN > %.1fN - stopping arm.",
                                     m, ceiling)
                        self._stop_arm()
                        abort = "watchdog"
                        break

                root_T_body = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME)
                travel = float(np.linalg.norm(
                    np.array([root_T_body.x, root_T_body.y]) - origin_xy))
                clearance, hits = self._clearance(goal)
                self._publish_clearance_viz(clearance, hits)
                gauge_ok, gauge, gauge_lim = self._recenter_gauge(st, snap)
                # Always the radial gauge, whatever _recenter_gauge is configured to
                # use: the floor is about the arm folding up against the body, which
                # is a length, not a conditioning number.
                reach_now = el.tip_reach_from_shoulder(snap, self._tip_from_hand)

                now = time.time()
                dt, last_tick = now - last_tick, now
                qm = get_a_tform_b(snap, root, WR1_FRAME_NAME).rot
                q_cmd = self._track_quat(goal, (qm.w, qm.x, qm.y, qm.z), q_cmd,
                                         start_elev, dt)
                # The tip is pinned in `root` here, so base yaw cannot move it: the
                # recenter steers on the raw aim error with no cumulative budget.
                aim_err, pivot, tip_live = self._aim_heading_err(goal, snap, root)
                self._publish_aim_viz(root, tip_live, None, pivot,
                                      self._body_x(root_T_body), throttle=True)
                v_rot = self._aim_yaw_rate(aim_err)
                mob = el.build_velocity_mobility_command(
                    self.recenter_body_vel, v_rot=v_rot,
                    max_lin_vel=self.recenter_body_vel,
                    max_ang_vel=max(self.aim_yaw_rate, 1e-3))
                # The hold point is fixed in `root`, but the command that carries it
                # is rooted in flat_body and re-derived every tick. The two are the
                # same physical target - flat_T_root moves the same world point into
                # the body frame - and the arm still folds, because the point walks
                # backwards through the body frame as the base advances.
                #
                # The difference is that the base actually walks. An ODOM-rooted arm
                # Cartesian command in the same SynchronizedCommand leaves the body
                # planted: measured over three recenter windows, forward body travel
                # along body +x was +0.001 m in 3.4 s (spot_push_policy_10 push 3, at
                # a commanded 0.10 m/s), and +0.001 / -0.001 m in bag
                # spot_push_policy_8 - against +0.549 m for the flat_body-rooted crawl
                # that runs straight afterwards. In _10 the feet never break contact
                # for the whole stage and the arm joints move under 3 deg; the stage
                # is a no-op that ends on its own timeout. Bag _8 only looked healthy
                # because its logged `travel` is a magnitude and was filled by 0.06 to
                # 0.19 m of SIDEWAYS drift, while the reach it exited on came from the
                # stale hold pose dragging the arm backwards - the artifact RECONTACT
                # removed.
                flat_T_root = get_a_tform_b(snap, GRAV_ALIGNED_BODY_FRAME_NAME, root)
                tip_flat = np.array(flat_T_root.transform_point(*tip_root))
                q_flat = self._root_quat_to_flat(snap, root, q_cmd)
                cmd = el.build_body_locked_arm_command(
                    tip_flat, q_flat,
                    remain_near_current_joints=False, duration_s=self.cmd_horizon,
                    tool_offset=self._tool_offset, mobility_command=mob)
                self.command_client.robot_command(
                    cmd, end_time_secs=now + self.topple_vel_end_time)

                track = self._track(goal)
                if track is not None and start_world is not None:
                    drift = float(np.linalg.norm(track[0] - start_world))
                self._publish_feedback(ExecutePushFeedback.PHASE_TOPPLE, latch.loaded,
                                       cur, drift)
                # The travel budget is live: whichever of the hard cap and the body
                # box clearance binds first.
                allowed = self.recenter_max_travel_cap
                if clearance is not None and np.isfinite(clearance):
                    allowed = min(allowed, travel + clearance)
                line = ("stage=recenter assist=topple %s=%.3f/%.3f reach=%.3f/%.3fm "
                        "travel=%.3f/%.3fm clr=%s drift=%.3fm F=%.1fN loaded=%d "
                        "low=%.2f/%.2fs elev=%+.1fdeg t=%.2f/%.2fs aim_err=%s "
                        "v_rot=%+.2frad/s"
                        % (self.reach_signal, gauge, gauge_lim, reach_now,
                           self.recenter_min_reach, travel, allowed,
                           self._clearance_str(clearance), drift, cur, latch.loaded,
                           latch.low_for, self.contact_lost_grace,
                           np.rad2deg(el.quat_elevation(q_cmd)), now - started,
                           self.recenter_timeout, self._aim_err_str(aim_err), v_rot))
                self.assist_pub.publish(String(line))
                if self.verbose:
                    rospy.loginfo_throttle(0.5, "ASSIST: %s%s", line,
                                           "" if latch.made else " [waiting for re-load]")

                # Aborts first: a lost contact means there is nothing to recenter
                # against, whatever the reach gauge says.
                if latch.lost:
                    abort = "toppled"
                    break
                if not latch.made and (now - entered) > self.topple_load_grace:
                    abort = "contact_lost"
                    break
                # ... then the exits that hand a (better) arm to the crawl.
                if gauge_ok:
                    reason = "gauge"
                    break
                if reach_now <= self.recenter_min_reach:
                    reason = "floor"
                    break
                if travel >= allowed:
                    reason = ("clearance" if (clearance is not None
                                              and np.isfinite(clearance)
                                              and allowed < self.recenter_max_travel_cap)
                              else "travel")
                    break
                if (now - started) > self.recenter_timeout:
                    reason = "timeout"
                    break
                rate.sleep()
        except Exception:
            rospy.logerr("Exception during recenter - stopping arm, no retry.",
                         exc_info=True)
            self._stop_arm()
            abort = "error"

        if abort is None and reason != "gauge":
            rospy.logwarn("ASSIST[topple]: recenter ended on %s, not on the reach "
                          "gauge: reach %.3f -> %.3fm (target %.3f), travel %.3fm, "
                          "clearance %s. The crawl still runs, with less arm "
                          "authority than it wanted.",
                          reason, reach_in, reach_now, self.recenter_target_reach,
                          travel, self._clearance_str(clearance))
        rospy.loginfo("ASSIST[topple]: recenter done reason=%s reach=%.3f->%.3fm "
                      "elev=%+.1f->%+.1fdeg travel=%.3fm clr=%s%s",
                      abort or reason, reach_in, reach_now, np.rad2deg(elev_in),
                      np.rad2deg(el.quat_elevation(q_cmd)), travel,
                      self._clearance_str(clearance),
                      "" if abort is None else " (ABORT)")
        return abort, peak, drift, snap, q_cmd

    def _track_quat(self, goal, meas_quat_wxyz, prev_cmd, ref_elev, dt):
        """
        Commanded tool orientation for the assist, in the push root frame: the task
        frame of the tracked push force, with the measured wrist roll kept, the
        elevation bounded to ref_elev +- recenter_pitch_band, and the whole step
        slew-limited to recenter_pitch_rate.

        force_world = R_object @ force_body, so this rotates WITH the object: as the
        object tips away, the commanded pitch follows the face down, which is what
        the arm-only push was already doing before the assist froze it.

        Roll is inherited rather than taken from the task frame because the push
        commands rx in force mode with zero torque - the real roll is compliant and
        the Gram-Schmidt roll was never a tracked setpoint. Making it one in an
        all-position command would snap the wrist while it is loaded.

        Falls back to the previous command (or the measurement, on the first tick)
        whenever the object track is unavailable, so a TF dropout freezes the
        orientation instead of dropping it.
        """
        track = self._track(goal)
        if track is None:
            return np.asarray(meas_quat_wxyz if prev_cmd is None else prev_cmd,
                              dtype=np.float64)
        q = el.task_quat_with_measured_roll(track[1], meas_quat_wxyz)
        q = el.clamp_quat_elevation(q, ref_elev - self.recenter_pitch_band,
                                    ref_elev + self.recenter_pitch_band)
        if prev_cmd is not None and self.recenter_pitch_rate > 0.0:
            q = el.slew_quat(prev_cmd, q, self.recenter_pitch_rate * max(dt, 0.0))
        return q

    def _root_quat_to_flat(self, snap, root, quat_root_wxyz):
        # Re-express a root-frame orientation in flat_body, so what recenter
        # commanded can be handed straight to the body-locked crawl. Both frames are
        # gravity-aligned, so this is a yaw-only change and elevation survives it.
        q = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME).rot
        R_root_flat = el.quat_wxyz_to_rotmat((q.w, q.x, q.y, q.z))
        return el.rotmat_to_quat_wxyz(
            R_root_flat.T @ el.quat_wxyz_to_rotmat(quat_root_wxyz))

    def _aim_heading_err(self, goal, snap, root):
        """
        (err, pivot, tip): the signed angle (rad) from the body's +x axis to the line
        from the LIVE tip to the object's yaw pivot. Positive means the base must yaw
        counterclockwise. err is None whenever there is nothing to aim at - no
        collision cloud, no object TF, aim_mode=push_dir, or a tip sitting over the
        pivot - and callers then fall back to the push axis and command no yaw.

        Measured from the TIP, not from the commanded contact point: the strut hands
        the legs' force to the object where it actually touches, and that is where
        the moment arm starts.
        """
        if self.aim_mode != "pivot":
            return None, None, None
        root_T_wr1 = get_a_tform_b(snap, root, WR1_FRAME_NAME)
        tip = np.array(root_T_wr1.transform_point(self.tool_tip_x, 0.0,
                                                  self.tool_tip_z))
        pivot = self._pivot_in_root(goal, root)
        if pivot is None:
            return None, None, tip
        r = pivot[:2] - tip[:2]
        if float(np.linalg.norm(r)) < self.aim_min_lever:
            return None, pivot, tip
        yaw = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME).rot.to_yaw()
        return el.wrap_pi(float(np.arctan2(r[1], r[0])) - yaw), pivot, tip

    def _aim_yaw_rate(self, aim_err, d_yaw=None):
        """
        Base yaw rate (rad/s) that steers the body back onto the aim, or 0.0.

        Proportional, rate-limited, with a deadband: the pivot is derived from a
        tracked pose and a sampled cloud, and without the deadband the base hunts
        around a few degrees of noise while it is loaded against the object.

        d_yaw (crawl only) is the yaw already spent since the tip was latched. Past
        aim_yaw_max the rate is zeroed in the direction that would spend more: the
        tip counter-rotation keeps the contact from sliding, but the tip still ends
        up further off the body x axis with every radian, and that is arm envelope
        the crawl is short of to begin with.
        """
        if not self.aim_track or aim_err is None:
            return 0.0
        if abs(aim_err) < self.aim_deadband:
            return 0.0
        rate = float(np.clip(self.aim_gain * aim_err,
                             -self.aim_yaw_rate, self.aim_yaw_rate))
        if d_yaw is not None and self.aim_yaw_max > 0.0:
            if (d_yaw >= self.aim_yaw_max and rate > 0.0) or \
               (d_yaw <= -self.aim_yaw_max and rate < 0.0):
                return 0.0
        return rate

    @staticmethod
    def _aim_err_str(aim_err):
        # "n/a" (nothing to aim at) or degrees, matching _clearance_str's contract.
        return "n/a" if aim_err is None else "%+.1fdeg" % np.rad2deg(aim_err)

    @staticmethod
    def _body_x(root_T_body):
        # Body +x as a root-frame xy direction; the reference the aim is drawn
        # against once the base is the thing being steered.
        yaw = root_T_body.rot.to_yaw()
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def _recenter_gauge(self, robot_state, snap):
        # (done, value, target): whether the arm has recovered enough envelope for
        # the crawl to lock onto it. The mirror of _reach_gauge, on the same signal:
        # that one says the arm is OUT of travel, this one says it is back in.
        if self.reach_signal == "manipulability":
            w = el.arm_manipulability(robot_state.kinematic_state.joint_states)
            return ((not np.isnan(w)) and w >= self.recenter_manip_min, w,
                    self.recenter_manip_min)
        reach = el.tip_reach_from_shoulder(snap, self._tip_from_hand)
        return reach <= self.recenter_target_reach, reach, self.recenter_target_reach

    def _clearance(self, goal):
        """
        (clearance_m, blocking_points) between Spot's body box and the object's
        collision cloud, both in flat_body. clearance is None when the test cannot
        run - no mesh or no object TF - so callers can tell "no data" from "+inf,
        nothing in the way" and fall back to the static travel cap.
        """
        if self.collision_pts is None:
            return None, None
        pose = self._object_pose_in_root(self._object_frame(goal),
                                         root=GRAV_ALIGNED_BODY_FRAME_NAME)
        if pose is None:
            return None, None
        R, t = pose
        return el.body_clearance_to_cloud(
            R, t, self.collision_pts, self.clr_half_len, self.clr_half_width,
            self.clr_half_height, front_margin=self.clr_front_margin,
            side_margin=self.clr_side_margin, over_margin=self.clr_over_margin,
            under_margin=self.clr_under_margin)

    @staticmethod
    def _clearance_str(clearance):
        # "n/a" (no mesh or no object TF), "inf" (nothing in the body's way at all,
        # e.g. an overhang it can walk under) or metres.
        if clearance is None:
            return "n/a"
        if not np.isfinite(clearance):
            return "inf"
        return "%.3fm" % clearance

    def _reach_gauge(self, robot_state, snap):
        # (near_limit, value, threshold): whether the arm is nearing its far-reach
        # singularity, plus the raw gauge value and the limit it is being compared
        # against, so the assist report can print both whichever signal is in use.
        # Two methods:
        # a) shoulder_radius: how far the pusher tip is from the SHOULDER PIVOT,
        #    radially. That is what the arm's reach spec measures, and the only
        #    cheap gauge that stays honest off the body x axis. (Body-frame tip x
        #    is not: it is offset by the shoulder's own 0.292m and drops the y/z
        #    extension, so it fires early on a level push and never on a low one.)
        # b) manipulability is the Yoshikawa index from the reported joint angles.
        # Not used by the retreat (retracting shortens the arm, so far-reach is never
        # the binding constraint there); kept as the reach gauge for the push phase.
        if self.reach_signal == "manipulability":
            w = el.arm_manipulability(robot_state.kinematic_state.joint_states)
            return (not np.isnan(w)) and w < self.manip_min, w, self.manip_min
        reach = el.tip_reach_from_shoulder(snap, self._tip_from_hand)
        return reach >= self.max_tip_reach, reach, self.max_tip_reach

    def _coordinated_retreat(self, goal, root, push_dir_world):
        """
        Retreat after a push: decay the contact force to zero, then pull the pusher tip
        straight back along -push_dir until it clears the object.

        The ARM owns the retraction: an all-axes POSITION command, reissued against the
        LIVE object pose so a tip that is still dragging the object can never be mistaken
        for a tip that has cleared it. The base stays planted, because retracting
        shortens the arm and far-reach is therefore never the binding constraint. It
        backs off only when the body-frame clamp binds, i.e. when the arm is genuinely
        out of travel.

        push_dir_world is the PUSH direction (into the object); the retreat runs along
        its negative.
        """
        push_dir = np.asarray(push_dir_world, dtype=np.float64)
        if np.linalg.norm(push_dir) < 1e-9:
            return
        push_dir = push_dir / np.linalg.norm(push_dir)
        # Freeze the retreat axis and hand orientation at entry. Re-deriving them
        # per-tick from the (ramping, near-zero) release force made
        # task_frame_from_force collapse to the identity task frame and snap the
        # wrist toward the body; hold push_dir and q_hold fixed for the whole retreat.
        _, q_hold = el.task_frame_from_force(push_dir)

        # The push loop leaves a base mobility command running, and an arm-only command
        # does NOT cancel it. Plant the base explicitly before the arm starts moving,
        # otherwise the stale mobility keeps driving the body while the arm retracts.
        stand_mob = el.build_stand_mobility_command()
        try:
            self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
        except Exception:
            rospy.logwarn("RETREAT: stand command failed; base may still be moving.",
                          exc_info=True)

        track = self._track(goal)
        contact_live = track[0] if track is not None else None
        if contact_live is None:
            rospy.logwarn("RETREAT: no object TF at entry, homing arm.")
            return

        # Bias-removed, like every other contact test: whether the tip is still
        # loaded decides between releasing a held force and a plain retract.
        m0, _ = self._contact_force(self.state_client.get_robot_state())
        state = "RETRACT" if (m0 is None or m0 < self.contact_eps) else "RELEASE"
        self._vlog("RETREAT: start state=%s (F0=%s).", state,
                   "n/a" if m0 is None else "%.1fN" % m0)

        rate = rospy.Rate(self.loop_rate)
        deadline = time.time() + self.retreat_deadline
        release_start = None
        retract_gated = False   # fire the pre-retract interactive gate only once
        misses = 0              # consecutive object-TF lookup failures
        base_backing = False    # True while the clamp binds and the base is receding
        last_base_cmd = 0.0     # throttle base SE2 reissue (see retract_cmd_period)
        clearance = 0.0

        try:
            while not rospy.is_shutdown():
                if time.time() >= deadline:
                    rospy.logwarn("RETREAT: deadline reached with tip-object %.3fm < "
                                  "%.3fm, homing arm.", clearance, self.safe_object_dist)
                    break

                st = self.state_client.get_robot_state()
                snap = st.kinematic_state.transforms_snapshot
                # Read out of the state already fetched this tick: a second
                # get_robot_state() would cost an RPC and report a different instant
                # than the clearance geometry it is compared against.
                m, raw = self._contact_force(st)
                cur = 0.0 if raw is None else raw
                self.force_pub.publish(Float32(cur))

                # Live object pose. Both the retract target and the done-check ride on
                # it; measuring against a pose frozen at entry lets the object's own
                # motion satisfy the clearance test while the tip never moves.
                track = self._track(goal)
                if track is None:
                    misses += 1
                    if misses > self.object_tf_max_misses:
                        rospy.logwarn("RETREAT: object TF lost (%d misses), homing arm.",
                                      misses)
                        break
                else:
                    misses = 0
                    contact_live = track[0]

                if state == "RELEASE":
                    if release_start is None:
                        release_start = time.time()
                    if self.release_time <= 0.0:
                        frac = 1.0
                    else:
                        frac = min(1.0, (time.time() - release_start) / self.release_time)
                    mag = (1.0 - frac) * self.retreat_hold_force
                    cmd, _, _ = el.build_arm_cartesian_command(
                        contact_live, push_dir * mag, root_frame=root,
                        duration_s=self.cmd_horizon, force_clip=self.force_clip,
                        task_quat=q_hold, tool_offset=self._tool_offset,
                        mobility_command=stand_mob)
                    self.command_client.robot_command(cmd)
                    contact_lost = m is not None and m < self.contact_eps
                    self._publish_feedback(ExecutePushFeedback.PHASE_RETREAT,
                                           not contact_lost, cur, 0.0)
                    if contact_lost or frac >= 1.0:
                        state = "RETRACT"
                        self._vlog("RETREAT: RELEASE -> RETRACT (%s, F=%.1fN).",
                                   "contact lost" if contact_lost else "released", cur)
                    rate.sleep()
                    continue

                # ------------------------------ RETRACT ----------------------- #
                # Contact is released here; pause before pulling the hand clear of the
                # object so a tester can watch/tune safe_object_dist.
                if not retract_gated:
                    self._gate("contact lost -> retract clear of object?")
                    retract_gated = True

                # measure the pusher tip (not the wrist/hand origin) vs the object.
                root_T_hand = get_a_tform_b(snap, root, HAND_FRAME_NAME)
                tip_now = np.array(root_T_hand.transform_point(*self._tip_from_hand))
                clearance = float(np.linalg.norm(tip_now - contact_live))
                if clearance >= self.safe_object_dist:
                    self._vlog("RETREAT: retract done, tip %.3fm >= safe_object_dist "
                               "%.3fm.", clearance, self.safe_object_dist)
                    break

                # Target the tip retreat_dist behind the live contact on the push axis,
                # then clamp it out of the body envelope.
                target = contact_live - self.retreat_dist * push_dir
                target, clamp_binds = el.clamp_target_to_body(
                    snap, root, target, self.min_hand_body_x)

                # Mobility is only re-sent on a transition or on the base throttle: an
                # arm-only command does not cancel a running mobility sub-command, so
                # the base holds whatever it was last told. Reissuing an SE2 goal at
                # loop_rate would supersede its step plan every tick and it would only
                # weight-shift, never walk.
                now = time.time()
                mob, mob_is_se2 = None, False
                if clamp_binds:
                    # Arm out of travel: the base has to open the room instead. Park it
                    # far enough behind the object that the tip lands clear of it.
                    body_T_hand = get_a_tform_b(snap, GRAV_ALIGNED_BODY_FRAME_NAME,
                                                HAND_FRAME_NAME)
                    tip_x = body_T_hand.transform_point(*self._tip_from_hand)[0]
                    base_xy = el.base_align_pose(contact_live, push_dir,
                                                 tip_x + self.safe_object_dist)
                    if base_xy is not None:   # None on a near-vertical push
                        if now - last_base_cmd >= self.retract_cmd_period:
                            mob = el.build_se2_mobility_command(
                                base_xy[0], base_xy[1], base_xy[2], root_frame=root,
                                max_lin_vel=self.retreat_base_max_lin_vel,
                                max_ang_vel=self.retreat_base_max_ang_vel)
                            mob_is_se2 = True
                            last_base_cmd = now
                        if not base_backing:
                            self._vlog("RETREAT: tip at body clamp %.2fm, base backing "
                                       "off (tip-object %.3fm).",
                                       self.min_hand_body_x, clearance)
                        base_backing = True
                elif base_backing:
                    mob = stand_mob         # clamp released -> re-plant the base
                    base_backing = False
                    self._vlog("RETREAT: clamp released, base re-planted.")

                cmd, _ = el.build_standoff_pose_command(
                    target, push_dir, q_hold, root_frame=root, eps=0.0,
                    duration_s=self.retract_move_time, tool_offset=self._tool_offset,
                    mobility_command=mob)
                if mob_is_se2:
                    # end_time_secs is the only thing that fills se2_trajectory_request
                    # .end_time; without it the goal lands already expired.
                    self.command_client.robot_command(
                        cmd, end_time_secs=now + self.se2_end_time)
                else:
                    self.command_client.robot_command(cmd)

                if self.verbose:
                    rospy.loginfo_throttle(
                        0.5, "RETREAT[retract]: tip-object %.3fm (target %.3fm)%s.",
                        clearance, self.safe_object_dist,
                        " [base backing off]" if base_backing else "")
                self._publish_feedback(ExecutePushFeedback.PHASE_RETREAT,
                                       False, cur, clearance)
                rate.sleep()
        except Exception:
            rospy.logerr("Exception during retreat - stopping arm.", exc_info=True)
            self._stop_arm()

    def _return_to_ready(self, goal, root, push_dir_world=None):
        # Home after a push: release the contact force, retract the tip clear of the
        # object along -push_dir, then arm to the carry pose. push_dir_world is the
        # last commanded push force direction (into the object), not a retreat vector.
        try:
            self._phase_mark("RETREAT")
            self._publish_feedback(ExecutePushFeedback.PHASE_RETREAT, False, 0.0, 0.0)
            if push_dir_world is not None:
                self._coordinated_retreat(goal, root, push_dir_world)
            self._go_to_carry_pose()
            rospy.loginfo("Arm returned to carry pose.")
        except Exception:
            rospy.logerr("Return-to-ready failed.", exc_info=True)

    def _go_to_carry_pose(self):
        # Move the arm to the body-frame carry (tuck) pose. Accepted directly from
        # a stowed arm, so no arm_ready/unstow is needed beforehand. Body-relative,
        # so it holds through base motion. Raises on failure for the caller to handle.
        #
        # Combine with an explicit stand: an arm-only synchro command does NOT
        # cancel a still-running mobility sub-command, so after the retreat the
        # base would keep moving while the arm homes. The stand halts the base and
        # holds it in place.
        carry_cmd = el.build_carry_pose_command(
            self.carry_hand_x, self.carry_hand_z,
            root_frame=GRAV_ALIGNED_BODY_FRAME_NAME,
            duration_s=self.carry_move_time,
            max_lin_vel=self.carry_max_lin_vel, max_accel=self.carry_max_accel,
            mobility_command=el.build_stand_mobility_command())
        carry_id = self.command_client.robot_command(carry_cmd)
        # Extra slack over carry_move_time: the rate caps can stretch the move past
        # the requested trajectory duration, so the wait must outlast the caps.
        block_until_arm_arrives(self.command_client, carry_id,
                                self.carry_move_time + 3.0)

    def _abort(self, end_reason, msg, peak=0.0, contact_made=False, drift=0.0):
        rospy.logwarn("Goal aborted: %s", msg)
        res = ExecutePushResult(contact_made=contact_made, peak_force=peak,
                                finger_travel=drift, end_reason=end_reason)
        self.server.set_aborted(res, msg)

    def _stop_arm(self):
        try:
            self.command_client.robot_command(RobotCommandBuilder.stop_command())
        except Exception:
            rospy.logerr("Stop command failed.", exc_info=True)

    def _safe_stop(self):
        if self.command_client is None:
            return
        try:
            self.command_client.robot_command(RobotCommandBuilder.stop_command())
        except Exception:
            pass

    def _stow_and_power_off(self):
        self.robot.logger.info("Stowing arm and powering off.")
        try:
            stow_id = self.command_client.robot_command(
                RobotCommandBuilder.arm_stow_command())
            block_until_arm_arrives(self.command_client, stow_id, 3.0)
        except Exception:
            self.robot.logger.exception("Stow failed; powering off anyway.")
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), "Robot power off failed."
        self.robot.logger.info("Safely powered off.")

    def _vlog(self, fmt, *args):
        # Verbose per-phase/subphase log, gated on ~exec/verbose.
        if self.verbose:
            rospy.loginfo(fmt, *args)

    def _gate(self, msg):
        # Block for Enter at a phase boundary when ~exec/interactive is set.
        # No-op otherwise. execute_cb runs on the action server's single worker
        # thread, so blocking here only stalls this goal (fine for bench tests).
        if not self.interactive:
            return
        try:
            input("[step] %s -- press Enter to continue... " % msg)
        except EOFError:
            # no stdin attached (e.g. launched, not a terminal); do not block
            rospy.logwarn("interactive gate: no stdin, continuing without pause")

    def _phase_mark(self, name):
        rospy.loginfo("PHASE -> %s", name)
        self.phase_pub.publish(UInt8(_PHASE_ENUM.get(name, 255)))
        if self._last_contact is not None:
            frame, pos = self._last_contact
            self._publish_text(frame, pos, name)
        if self.phase_pause > 0.0:
            rospy.sleep(self.phase_pause)

    def _publish_feedback(self, phase, contact_made, cur_force, point_drift):
        fb = ExecutePushFeedback()
        fb.phase = phase
        fb.contact_made = bool(contact_made)
        fb.cur_force = float(cur_force)
        fb.point_drift = float(point_drift)
        self.server.publish_feedback(fb)
        self.phase_pub.publish(UInt8(phase))

    def _publish_sphere(self, frame, pos, mid, rgba, diam=0.05):
        self._last_contact = (frame, list(pos))
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = rospy.Time.now()
        m.ns = "npm_push"
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = Point(*[float(v) for v in pos])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = diam
        m.color = ColorRGBA(*rgba)
        m.lifetime = rospy.Duration(0.0)
        self.marker_pub.publish(m)

    def _publish_standoff_pose(self, frame, pos, quat_wxyz):
        # Commanded fingertip standoff pose (position + orientation) for rviz.
        # quat_wxyz is the task/push orientation; PoseStamped is xyzw.
        p = PoseStamped()
        p.header.frame_id = frame
        p.header.stamp = rospy.Time.now()
        p.pose.position = Point(*[float(v) for v in pos])
        p.pose.orientation.w = float(quat_wxyz[0])
        p.pose.orientation.x = float(quat_wxyz[1])
        p.pose.orientation.y = float(quat_wxyz[2])
        p.pose.orientation.z = float(quat_wxyz[3])
        self.standoff_pub.publish(p)

    def _publish_aim_viz(self, frame, contact, aim, pivot, reference, throttle=False):
        """
        Draw the aim geometry: the yaw pivot, the contact->pivot line the base is
        steered onto, and the reference direction it is being compared against (the
        push axis at ALIGN, the body x axis during the assist).

        The number that matters is the ANGLE between the last two, and no log line
        shows an angle. aim=None derives it from contact and pivot.
        """
        if throttle:
            now = time.time()
            if self.clr_viz_rate <= 0.0 or (now - self._aim_viz_last) < (
                    1.0 / self.clr_viz_rate):
                return
            self._aim_viz_last = now
        if contact is None:
            return
        c = np.asarray(contact, dtype=np.float64).reshape(3)

        def _line(mid, tip, rgba):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = rospy.Time.now()
            m.ns = "npm_aim"
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.012
            m.color = ColorRGBA(*rgba)
            m.points = [Point(float(c[0]), float(c[1]), float(c[2])),
                        Point(float(tip[0]), float(tip[1]), float(tip[2]))]
            m.lifetime = rospy.Duration(0.0)
            self.aim_pub.publish(m)

        if pivot is not None:
            pv = np.asarray(pivot, dtype=np.float64).reshape(3)
            sph = Marker()
            sph.header.frame_id = frame
            sph.header.stamp = rospy.Time.now()
            sph.ns = "npm_aim"
            sph.id = 0
            sph.type = Marker.SPHERE
            sph.action = Marker.ADD
            sph.pose.position = Point(float(pv[0]), float(pv[1]), float(pv[2]))
            sph.pose.orientation.w = 1.0
            sph.scale.x = sph.scale.y = sph.scale.z = 0.07
            sph.color = ColorRGBA(1.0, 0.0, 1.0, 0.9)
            sph.lifetime = rospy.Duration(0.0)
            self.aim_pub.publish(sph)
            _line(1, [pv[0], pv[1], c[2]], (0.0, 0.9, 0.9, 0.9))
        elif aim is not None:
            a = np.asarray(aim, dtype=np.float64).reshape(-1)
            _line(1, [c[0] + 0.6 * a[0], c[1] + 0.6 * a[1], c[2]],
                  (0.0, 0.9, 0.9, 0.9))
        if reference is not None:
            d = np.asarray(reference, dtype=np.float64).reshape(-1)[:2]
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                d = d / n
                _line(2, [c[0] + 0.6 * d[0], c[1] + 0.6 * d[1], c[2]],
                      (1.0, 0.5, 0.0, 0.9))

    def _publish_clearance_viz(self, clearance, hits):
        """
        Draw the body box and the object points that block it, in flat_body.

        Throttled to ~clearance/viz_rate: the gate runs at loop_rate, but rviz does
        not need 40 Hz of a box that moves with the body it is drawn in. The box is
        published even when the test cannot run (grey), because "no clearance data"
        is exactly the state that is otherwise invisible.
        """
        now = time.time()
        if self.clr_viz_rate <= 0.0 or (now - self._clr_viz_last) < (1.0 / self.clr_viz_rate):
            return
        self._clr_viz_last = now
        # One namespace per topic. Both displays are rviz/Marker, and a shared
        # namespace gives them the same checkbox label in the display tree.
        frame = GRAV_ALIGNED_BODY_FRAME_NAME
        # The drawn box is the tested box: half extents grown by the same margins
        # body_clearance_to_cloud uses, so what you see is what blocks.
        x_lo, x_hi = -self.clr_half_len, self.clr_half_len + self.clr_front_margin
        y_half = self.clr_half_width + self.clr_side_margin
        z_lo = -(self.clr_half_height + self.clr_under_margin)
        z_hi = self.clr_half_height + self.clr_over_margin
        if clearance is None:
            rgba = (0.6, 0.6, 0.6, 0.25)
        elif clearance > 0.0:
            rgba = (0.1, 0.9, 0.1, 0.25)
        else:
            rgba = (0.9, 0.1, 0.1, 0.35)

        box = Marker()
        box.header.frame_id = frame
        box.header.stamp = rospy.Time.now()
        box.ns = "npm_body_box"
        box.id = 0
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position = Point(0.5 * (x_lo + x_hi), 0.0, 0.5 * (z_lo + z_hi))
        box.pose.orientation.w = 1.0
        box.scale.x = x_hi - x_lo
        box.scale.y = 2.0 * y_half
        box.scale.z = z_hi - z_lo
        box.color = ColorRGBA(*rgba)
        box.lifetime = rospy.Duration(0.0)
        self.body_box_pub.publish(box)

        txt = Marker()
        txt.header.frame_id = frame
        txt.header.stamp = rospy.Time.now()
        txt.ns = "npm_body_box"
        txt.id = 1
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position = Point(x_hi, 0.0, z_hi + 0.15)
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.08
        txt.color = ColorRGBA(1.0, 1.0, 0.1, 1.0)
        txt.text = "clearance %s" % self._clearance_str(clearance)
        txt.lifetime = rospy.Duration(0.0)
        self.body_box_pub.publish(txt)

        pts = Marker()
        pts.header.frame_id = frame
        pts.header.stamp = rospy.Time.now()
        pts.ns = "npm_clearance_pts"
        pts.id = 2
        pts.type = Marker.POINTS
        pts.action = Marker.ADD
        pts.pose.orientation.w = 1.0
        pts.scale.x = pts.scale.y = 0.01
        pts.color = ColorRGBA(1.0, 0.4, 0.0, 1.0)
        if hits is not None and len(hits):
            # Subsample: the blocking set can be most of a 2000-point cloud, and a
            # marker that large at viz_rate is a lot of rviz for no extra insight.
            step = max(1, len(hits) // 500)
            pts.points = [Point(float(p[0]), float(p[1]), float(p[2]))
                          for p in hits[::step]]
        pts.lifetime = rospy.Duration(0.0)
        self.clearance_pub.publish(pts)

    def _publish_text(self, frame, pos, text):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = rospy.Time.now()
        m.ns = "npm_push"
        m.id = 3
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = Point(float(pos[0]), float(pos[1]), float(pos[2]) + 0.3)
        m.pose.orientation.w = 1.0
        m.scale.z = 0.1     # text height
        m.color = ColorRGBA(1.0, 1.0, 0.1, 1.0)
        m.text = text
        m.lifetime = rospy.Duration(0.0)
        self.marker_pub.publish(m)

    def _publish_marker(self, frame, contact, force, fmag):
        self._last_contact = (frame, list(contact))
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = rospy.Time.now()
        m.ns = "npm_push"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        scale = (self.arrow_len * min(fmag, el.FORCE_NORM_CLIP) / el.FORCE_NORM_CLIP) \
            / max(fmag, 1e-6)
        end = [contact[i] + scale * force[i] for i in range(3)]
        m.points = [Point(*[float(v) for v in contact]), Point(*end)]
        m.scale.x = 0.01    # shaft diameter
        m.scale.y = 0.02    # head diameter
        m.scale.z = 0.0
        m.color = ColorRGBA(0.1, 1.0, 0.1, 1.0)
        m.lifetime = rospy.Duration(0.0)
        self.marker_pub.publish(m)


def main():
    rospy.init_node("executor_node")
    node = ExecutorNodeReal()
    node.run()


if __name__ == "__main__":
    main()

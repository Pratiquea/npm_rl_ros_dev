#!/usr/bin/env python3
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


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

sys.path.insert(0, os.path.join(_HERE, "..", "..", "npm_policy", "src"))
from npm_control import executor_lib as el
from npm_policy import obs_lib as ol


_PHASE_ENUM = {"ALIGN": ExecutePushFeedback.PHASE_ALIGN,
               "APPROACH": ExecutePushFeedback.PHASE_APPROACH,
               "PUSH": ExecutePushFeedback.PHASE_PUSH,
               "RETREAT": ExecutePushFeedback.PHASE_RETREAT,
               "TOPPLE": ExecutePushFeedback.PHASE_TOPPLE}


def _ee_force_vec_from_state(st):
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
        self.object_frame = rospy.get_param("~exec/object_frame", "object_link")
        self.npz_path = rospy.get_param("~npz_path", "")
        self.pcl_shrink = float(rospy.get_param("~pcl_shrink", 1.0))
        self.push_point_tol = rospy.get_param("~exec/push_point_tol", 0.005)
        self.link_pts = None
        self._push_point = None
        self._point_src = "none"

        self._force_bias = np.zeros(3, dtype=np.float64)
        self.dry_run = rospy.get_param("~exec/dry_run", True)
        self.force_clip = rospy.get_param("~exec/force_clip", el.FORCE_NORM_CLIP)
        self.watchdog_margin = rospy.get_param("~exec/watchdog_margin", 15.0)
        self.arrow_len = rospy.get_param("~viz/arrow_len", 1.0)
        self.loop_rate = rospy.get_param("~exec/loop_rate", 40.0)
        self.cmd_horizon = rospy.get_param("~exec/cmd_horizon", 0.6)
        self.tf_timeout = rospy.get_param("~exec/tf_timeout", 0.2)

        self.se2_end_time = rospy.get_param("~exec/se2_end_time", 2.0)
        self.phase_pause = rospy.get_param("~exec/phase_pause", 0.0)
        self.verbose = rospy.get_param("~exec/verbose", True)
        self.interactive = rospy.get_param("~exec/interactive", False)
        self.tool_tip_x = rospy.get_param("~exec/tool_tip_x", 0.24)
        self.tool_tip_z = rospy.get_param("~exec/tool_tip_z", 0.03)
        self._tool_offset = (self.tool_tip_x, self.tool_tip_z)
        self._tip_from_hand = np.array(
            [self.tool_tip_x - el._SPOT_ARM_TOOL_OFFSET[0], 0.0,
             self.tool_tip_z - el._SPOT_ARM_TOOL_OFFSET[2]], dtype=np.float64)
        self.time_sync_timeout = rospy.get_param("~exec/time_sync_timeout", 15.0)
        self.time_sync_retries = rospy.get_param("~exec/time_sync_retries", 3)
        self.max_finger_reach = rospy.get_param("~max_finger_reach", 0.6)

        self.align_standoff = rospy.get_param("~align/align_standoff", 0.15)
        self.base_standoff = rospy.get_param("~align/base_standoff", 0.73)
        self.base_walk_timeout = rospy.get_param("~align/base_walk_timeout", 10.0)
        self.base_max_lin_vel = rospy.get_param("~align/base_max_lin_vel", 0.5)
        self.base_max_ang_vel = rospy.get_param("~align/base_max_ang_vel", 0.75)
        self.carry_hand_x = rospy.get_param("~align/carry_hand_x", 0.47)
        self.carry_hand_z = rospy.get_param("~align/carry_hand_z", 0.41)
        self.carry_move_time = rospy.get_param("~align/carry_move_time", 2.5)
        self.carry_max_lin_vel = rospy.get_param("~align/carry_max_lin_vel", 0.25)
        self.carry_max_accel = rospy.get_param("~align/carry_max_accel", 0.5)
        self.aim_mode = rospy.get_param("~align/aim_mode", "pivot")
        self.aim_cone = np.deg2rad(rospy.get_param("~align/aim_cone", 35.0))
        self.aim_support_eps = rospy.get_param("~align/support_eps", 0.015)
        self.aim_min_lever = rospy.get_param("~align/min_lever", 0.05)

        self.approach_vel = rospy.get_param("~approach/approach_vel", 0.05)
        self.approach_accel = rospy.get_param("~approach/approach_accel", 0.25)
        self.approach_overshoot = rospy.get_param("~approach/approach_overshoot", 0.02)
        self.approach_timeout = rospy.get_param("~approach/approach_timeout", 8.0)
        self.bias_window = rospy.get_param("~approach/bias_window", 0.3)
        self.approach_stall_window = rospy.get_param("~approach/stall_window", 0.5)
        self.approach_stall_eps = rospy.get_param("~approach/stall_eps", 0.008)
        self.approach_force_ceiling = rospy.get_param("~approach/force_ceiling", 30.0)

        self.contact_eps = rospy.get_param("~push/contact_eps", 2.0)
        self.contact_made_n = rospy.get_param("~push/contact_made_n", 5.0)
        if rospy.has_param("~push/approach_force"):
            rospy.logwarn("~push/approach_force is OBSOLETE and IGNORED; the approach "
                          "is now a speed-limited position move (~approach/approach_vel). "
                          "Use ~push/push_force_start to seed the push ramp.")
        self.push_force_start = rospy.get_param("~push/push_force_start",
                                                self.contact_made_n)
        self.contact_lost_grace = rospy.get_param("~push/contact_lost_grace", 0.4)
        self.ramp_time = rospy.get_param("~push/ramp_time", 1.0)
        self.tilt_root_frame = rospy.get_param("~topple/tilt_root_frame", "world")

        self.topple_tilt_min_deg = rospy.get_param("~topple/tilt_min_deg", 30.0)
        self.topple_tilt_min = np.deg2rad(self.topple_tilt_min_deg)
        if rospy.has_param("~topple/tilt_min"):
            rospy.logwarn("~topple/tilt_min (radians) is DEPRECATED and IGNORED; using "
                          "~topple/tilt_min_deg=%.2fdeg (%.3frad). Remove the old key.",
                          self.topple_tilt_min_deg, self.topple_tilt_min)
        self.reach_assist = rospy.get_param("~topple/reach_assist", True)
        self.topple_body_vel = rospy.get_param("~topple/body_vel", 0.15)
        self.topple_max_body_travel = rospy.get_param("~topple/max_body_travel", 0.5)
        self.reach_max_body_travel = rospy.get_param("~topple/reach_max_body_travel", 0.5)
        self.topple_stall_window = rospy.get_param("~topple/stall_window", 0.5)
        self.topple_stall_eps = rospy.get_param("~topple/stall_eps", 0.01)
        self.topple_load_grace = rospy.get_param("~topple/load_grace", 1.0)
        self.topple_vel_end_time = rospy.get_param("~topple/vel_end_time", 0.5)

        self.topple_duration_override = rospy.get_param("~topple/duration_override", True)
        self.topple_max_beyond_duration = rospy.get_param(
            "~topple/max_time_beyond_duration_override", 4.0)
        self.topple_tilt_rise_eps_deg = rospy.get_param("~topple/tilt_rise_eps_deg", 0.5)
        self.topple_tilt_rise_eps = np.deg2rad(self.topple_tilt_rise_eps_deg)


        self.recontact = rospy.get_param("~topple/recontact", True)
        self.recontact_wait = rospy.get_param("~topple/recontact_wait", 2.5)
        self.recontact_settle = rospy.get_param("~topple/recontact_settle", 0.5)

        self.recenter = rospy.get_param("~topple/recenter", True)
        self.recenter_target_reach = rospy.get_param("~topple/recenter_target_reach",
                                                     0.75)
        self.recenter_manip_min = rospy.get_param("~topple/recenter_manip_min", 0.05)
        self.recenter_min_reach = rospy.get_param("~topple/recenter_min_reach", 0.55)
        self.recenter_max_travel_cap = rospy.get_param(
            "~topple/recenter_max_travel_cap", 0.30)
        self.recenter_timeout = rospy.get_param("~topple/recenter_timeout", 3.0)
        self.recenter_body_vel = rospy.get_param("~topple/recenter_body_vel", 0.10)
        self.recenter_track_orientation = rospy.get_param(
            "~topple/recenter_track_orientation", True)
        self.recenter_pitch_band_deg = rospy.get_param("~topple/recenter_pitch_band",
                                                       25.0)
        self.recenter_pitch_band = np.deg2rad(self.recenter_pitch_band_deg)
        self.recenter_pitch_rate_deg = rospy.get_param("~topple/recenter_pitch_rate",
                                                       30.0)
        self.recenter_pitch_rate = np.deg2rad(self.recenter_pitch_rate_deg)
        self.recenter_clearance_gate = rospy.get_param(
            "~topple/recenter_clearance_gate", True)
        self.aim_track = rospy.get_param("~topple/aim_track", True)
        self.aim_yaw_rate = np.deg2rad(rospy.get_param("~topple/aim_yaw_rate", 10.0))
        self.aim_gain = rospy.get_param("~topple/aim_gain", 1.0)
        self.aim_deadband = np.deg2rad(rospy.get_param("~topple/aim_deadband", 5.0))
        self.aim_yaw_max = np.deg2rad(rospy.get_param("~topple/aim_yaw_max", 20.0))

        self.clr_samples = int(rospy.get_param("~clearance/samples", 2000))
        self.clr_bbox_tol = rospy.get_param("~clearance/bbox_tol", 0.005)
        self.clr_half_len = rospy.get_param("~clearance/half_len", 0.55)
        self.clr_half_width = rospy.get_param("~clearance/half_width", 0.25)
        self.clr_half_height = rospy.get_param("~clearance/half_height", 0.10)
        self.clr_front_margin = rospy.get_param("~clearance/front_margin", 0.12)
        self.clr_side_margin = rospy.get_param("~clearance/side_margin", 0.05)
        self.clr_over_margin = rospy.get_param("~clearance/over_margin", 0.05)
        self.clr_under_margin = rospy.get_param("~clearance/under_margin", 0.05)
        self.clr_viz_rate = rospy.get_param("~clearance/viz_rate", 5.0)
        self._clr_viz_last = 0.0
        self._aim_viz_last = 0.0
        self.collision_pts = None

        self.min_hand_body_x = rospy.get_param("~retreat/min_hand_body_x", 0.55)
        self.reach_signal = rospy.get_param("~retreat/reach_signal", "shoulder_radius")
        self.max_tip_reach = rospy.get_param("~retreat/max_tip_reach", 0.93)
        self.manip_min = rospy.get_param("~retreat/manip_min", 0.02)
        self.retreat_hold_force = rospy.get_param("~retreat/retreat_hold_force", 10.0)
        self.release_time = rospy.get_param("~retreat/release_time", 2.0)
        self.safe_object_dist = rospy.get_param("~retreat/safe_object_dist", 0.25)
        self.retreat_dist = rospy.get_param("~retreat/retreat_dist", 0.30)
        self.retract_move_time = rospy.get_param("~retreat/retract_move_time", 1.0)
        self.object_tf_max_misses = rospy.get_param("~retreat/object_tf_max_misses", 20)
        self.retract_cmd_period = rospy.get_param("~retreat/retract_cmd_period", 0.4)
        self.retreat_deadline = rospy.get_param("~retreat/retreat_deadline", 15.0)
        self.retreat_base_max_lin_vel = rospy.get_param("~retreat/base_max_lin_vel", 0.5)
        self.retreat_base_max_ang_vel = rospy.get_param("~retreat/base_max_ang_vel", 0.75)

        self._load_object_model()
        self.model_checked = False
        rospy.Subscriber("/npm/model_info", String, self._model_cb, queue_size=1)
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

        self._last_contact = None
        self.marker_pub = rospy.Publisher("/npm/push_marker", Marker, queue_size=1)
        self.force_pub = rospy.Publisher("/npm/ee_force", Float32, queue_size=1)
        self.phase_pub = rospy.Publisher("/npm/executor/phase", UInt8, queue_size=1,
                                         latch=True)
        self.assist_pub = rospy.Publisher("/npm/executor/assist_debug", String,
                                          queue_size=1)
        self.body_box_pub = rospy.Publisher("/npm/executor/body_box", Marker,
                                            queue_size=2)
        self.clearance_pub = rospy.Publisher("/npm/executor/clearance_pts", Marker,
                                             queue_size=1)
        self.aim_pub = rospy.Publisher("/npm/executor/aim", Marker, queue_size=3)
        self.standoff_pub = rospy.Publisher("/npm/standoff_pose", PoseStamped,
                                            queue_size=1, latch=True)
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
        self._wait_for_time_sync()

        assert self.robot.has_arm(), "Robot requires an arm to run this test."
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

            self.robot.logger.info("Moving arm to carry pose...")
            self._go_to_carry_pose()

            rospy.on_shutdown(self._safe_stop)
            self.server.start()  
            self.robot.logger.info("Ready - ExecutePush action server up.")
            try:
                rospy.spin()
            finally:
                self._stow_and_power_off()

    def _load_object_model(self):
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
        if self.model_checked or not self.npz_path:
            return
        rospy.logwarn("no /npm/model_info after %.1f s: object model UNVERIFIED. "
                      "loc_idx goals still resolve, but nothing has confirmed the "
                      "publisher holds the same .npz and pcl_shrink=%.4f.",
                      self.model_timeout, self.pcl_shrink)

    def _resolve_push_point(self, goal):
        try:
            point, src = el.resolve_push_point(goal.loc_idx, goal.push_point,
                                               self.link_pts)
        except ValueError as exc:
            self._abort("error", str(exc))
            return False
        self._push_point, self._point_src = point, src
        if src == "loc_idx":
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
        return goal.object_frame or self.object_frame

    def _tilt_rotation(self, object_frame, tilt_root):
        pose = self._object_pose_in_root(object_frame, root=tilt_root)
        return None if pose is None else pose[0]

    def _track(self, goal):
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
        self._clr_viz_last = 0.0
        self._publish_clearance_viz(*self._clearance(goal))
        self._aim_viz_last = 0.0
        self._publish_aim_viz(root, contact_world, None,
                              self._pivot_in_root(goal, root), push_dir)

        if self.dry_run:
            res = ExecutePushResult(contact_made=False, peak_force=0.0,
                                    finger_travel=0.0, end_reason="dry_run")
            self.server.set_succeeded(res)
            return

        self._phase_mark("ALIGN")
        self._publish_feedback(ExecutePushFeedback.PHASE_ALIGN, False, 0.0, 0.0)
        self._publish_sphere(root, contact_world, mid=1, rgba=(0.1, 1.0, 0.1, 0.9))
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

        self._vlog("ALIGN[base]: walking base behind contact along push_dir "
                   "(standoff=%.2fm).", self.base_standoff + self.align_standoff)
        if not self._walk_base_align(goal, contact_world, push_dir, root):
            self._stop_arm()
            self._abort("error", "align failed")
            return
        self._vlog("ALIGN[base]: base in position.")
        self._gate("base aligned -> arm cartesian align?")

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
        self._measure_force_bias()
        approach_reason, latch, approach_peak = self._approach_loop(goal, root, q)
        if approach_reason == "error":
            self._abort("error", "approach exception", peak=approach_peak,
                        contact_made=latch.made)
            rospy.signal_shutdown("approach exception")
            return
        self._push_loop(goal, root, latch, approach_peak, approach_reason)

    def _measure_force_bias(self):
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
        v = _ee_force_vec_from_state(st)
        if v is None:
            return None, None
        return (float(np.linalg.norm(v - self._force_bias)),
                float(np.linalg.norm(v)))

    def _approach_loop(self, goal, root, task_quat):
        latch = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                self.contact_lost_grace)
        peak = 0.0
        ceiling = self.approach_force_ceiling
        stall_gate = (self.approach_stall_window
                      + self.approach_vel / max(self.approach_accel, 1e-6))
        rate = rospy.Rate(self.loop_rate)
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
        return "preempted", latch, peak

    def _pivot_in_root(self, goal, root):
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
        aim, pivot, ang = self._align_aim(goal, contact_world, push_dir, root)
        goal_xy = None if aim is None else el.base_align_pose(
            contact_world, [aim[0], aim[1], 0.0],
            self.base_standoff + self.align_standoff)
        if goal_xy is None:
            rospy.logwarn("ALIGN: push near-vertical, skipping base pre-walk.")
            return True
        bx, by, yaw = goal_xy
        self._publish_aim_viz(root, contact_world, aim, pivot, push_dir)
        self._vlog("ALIGN[base]: target x=%.3f y=%.3f yaw=%.3frad in %s.",
                   bx, by, yaw, root)
        try:
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
        start = self._track(goal)
        start_world = start[0] if start is not None else None
        start_elev = None
        if start is not None:
            start_elev = el.quat_elevation(el.task_frame_from_force(start[1])[1])
        object_frame = self._object_frame(goal)
        tilt_root = self.tilt_root_frame or root
        tilt_ref = self._tilt_rotation(object_frame, tilt_root)
        if tilt_ref is None and tilt_root != root:
            rospy.logwarn("tilt root %s unavailable at push start; falling back to %s. "
                          "Tilt will lag the object by the driver's TF latency.",
                          tilt_root, root)
            tilt_root = root
            tilt_ref = self._tilt_rotation(object_frame, tilt_root)
        peak = 0.0
        ceiling = self.force_clip + self.watchdog_margin
        rate = rospy.Rate(self.loop_rate)
        deadline = None
        if latch.made_time is not None:
            deadline = latch.made_time + self.ramp_time + float(goal.duration)
        goal_deadline_init = deadline
        end_reason, drift = "timeout", 0.0
        last_force_world = None
        push_dir = None
        stand_mob = el.build_stand_mobility_command()
        tip_hist = []
        tilt_hist = []
        waived = 0.0
        goal_deadline = goal_deadline_init
        hard_cap = (None if goal_deadline_init is None
                    else goal_deadline_init + self.topple_max_beyond_duration)
        verdict = None
        tilt_gain = 0.0

        if approach_reason is not None:
            end_reason = approach_reason
        else:
            self.phase_pub.publish(UInt8(ExecutePushFeedback.PHASE_PUSH))
            rospy.loginfo("PHASE -> PUSH (contact made, approach peak %.1fN); macro "
                          "clock armed for %.2fs ramp + %.2fs push",
                          approach_peak, self.ramp_time, goal.duration)

        try:
            while approach_reason is None and not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    end_reason = "preempted"
                    break

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
                    cmd, _, _ = el.build_arm_cartesian_command(
                        contact_world, force_cmd, root_frame=root,
                        duration_s=self.cmd_horizon, force_clip=self.force_clip,
                        tool_offset=self._tool_offset, mobility_command=stand_mob)
                    self.command_client.robot_command(cmd)
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

                arm_out, tip_reach, reach_lim = self._reach_gauge(st, snap)
                stalled, tip_adv = self._tip_stalled(tip_hist)
                if (latch.made_time is None
                        or (time.time() - latch.made_time)
                        < (self.ramp_time + self.topple_stall_window)):
                    stalled, tip_adv = False, float("nan")
                drift_out = drift > self.max_finger_reach
                out_of_travel = arm_out or drift_out or stalled
                loaded = latch.loaded
                toppling = tilt_gain > self.topple_tilt_min and loaded
                mode = None
                if latch.made and out_of_travel and push_dir is not None:
                    if toppling:
                        mode = "topple"
                    elif self.reach_assist and not drift_out:
                        mode = "reach"
                verdict = self._assist_report(
                    mode=mode, tilt_gain=tilt_gain, force=m, contact_made=latch.made,
                    arm_out=arm_out, tip_reach=tip_reach, reach_lim=reach_lim,
                    drift=drift, drift_out=drift_out, stalled=stalled,
                    tip_adv=tip_adv, out_of_travel=out_of_travel, toppling=toppling,
                    loaded=loaded, low_for=latch.low_for, waived=waived)
                if mode is not None:
                    end_reason, peak, drift = self._body_assist(
                        goal, root, snap, push_dir, start_world, peak, drift, ceiling,
                        mode=mode, deadline=deadline, start_elev=start_elev)
                    break
                if arm_out:
                    end_reason = "arm_limit"
                    break

                if drift > self.max_finger_reach:
                    end_reason = "reach"
                    break
                if latch.lost:
                    end_reason = "contact_lost"
                    break
                if deadline is not None and time.time() >= deadline:
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
            rospy.logerr("Exception during push - stopping arm, no retry.", exc_info=True)
            self._stop_arm()
            self._abort("error", "push exception", peak=peak,
                        contact_made=latch.made, drift=drift)
            rospy.signal_shutdown("push exception")
            return

        if last_force_world is not None:
            track = self._track(goal)
            if track is not None:
                self._publish_marker(root, track[0], last_force_world,
                                     float(min(np.linalg.norm(last_force_world),
                                               self.force_clip)))

        res = ExecutePushResult(contact_made=latch.made, peak_force=peak,
                                finger_travel=drift, end_reason=end_reason)
        rospy.loginfo("PushResult: contact=%s peak=%.1fN (approach peak %.1fN) "
                      "drift=%.3fm reason=%s%s",
                      latch.made, peak, approach_peak, drift, end_reason,
                      "" if waived <= 0.0
                      else " (duration override +%.1fs)" % waived)
        rospy.loginfo("ASSIST verdict: %s",
                      verdict if verdict is not None else "no push tick completed")
        self._return_to_ready(goal, root, last_force_world)
        if end_reason == "preempted":
            self.server.set_preempted(res)
        else:
            self.server.set_succeeded(res)

    def _sample_series(self, hist, value, window=None):
        window = self.topple_stall_window if window is None else window
        now = time.time()
        hist.append((now, value))
        cutoff = now - window
        while len(hist) > 2 and hist[1][0] < cutoff:
            hist.pop(0)

    def _tip_stalled(self, hist, window=None, eps=None):
        window = self.topple_stall_window if window is None else window
        eps = self.topple_stall_eps if eps is None else eps
        if len(hist) < 2 or (hist[-1][0] - hist[0][0]) < window:
            return False, float("nan")
        adv = hist[-1][1] - hist[0][1]
        return adv < eps, adv

    def _assist_report(self, mode, tilt_gain, force, contact_made, arm_out, tip_reach,
                       reach_lim, drift, drift_out, stalled, tip_adv, out_of_travel,
                       toppling, loaded, low_for, waived):
        tilt_ok = tilt_gain > self.topple_tilt_min
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
        self._phase_mark("TOPPLE")
        latch = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                self.contact_lost_grace)
        entered = time.time()
        if start_elev is None:
            root_T_wr1 = get_a_tform_b(snap, root, WR1_FRAME_NAME)
            qm = root_T_wr1.rot
            start_elev = el.quat_elevation((qm.w, qm.x, qm.y, qm.z))
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
            abort, peak, drift, snap, tip_root, q_cmd_root, _why = \
                self._await_recontact(goal, root, peak, drift, ceiling, start_world,
                                      start_elev)
            if abort is not None:
                self._plant_base(mode)
                rospy.loginfo("ASSIST[%s] done: reason=%s (during recontact) "
                              "drift=%.3fm peak=%.1fN", mode, abort, drift, peak)
                return abort, peak, drift
            latch = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                    self.contact_lost_grace)
            latch.force_made(time.time())
            entered = time.time()

        if mode == "topple" and self.recenter:
            abort, peak, drift, snap, q_cmd_root = self._recenter(
                goal, root, snap, peak, drift, ceiling, latch, entered, start_world,
                start_elev, tip_root=tip_root, q_cmd=q_cmd_root)
            if abort is not None:
                self._plant_base(mode)
                rospy.loginfo("ASSIST[%s] done: reason=%s (during recenter) "
                              "drift=%.3fm peak=%.1fN", mode, abort, drift, peak)
                return abort, peak, drift

        flat_T_wr1 = get_a_tform_b(snap, GRAV_ALIGNED_BODY_FRAME_NAME, WR1_FRAME_NAME)
        tip_body = flat_T_wr1.transform_point(self.tool_tip_x, 0.0, self.tool_tip_z)
        q_body = flat_T_wr1.rot
        quat_wxyz = (q_body.w, q_body.x, q_body.y, q_body.z)
        if q_cmd_root is not None:
            quat_wxyz = self._root_quat_to_flat(snap, root, q_cmd_root)
        root_T_body = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME)
        origin_xy = np.array([root_T_body.x, root_T_body.y])
        yaw0 = root_T_body.rot.to_yaw()

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
                        rospy.logerr("WATCHDOG: EE force %.1fN > %.1fN - stopping arm.",
                                     m, ceiling)
                        self._stop_arm()
                        end_reason = "watchdog"
                        break

                root_T_body = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME)
                travel = float(np.linalg.norm(
                    np.array([root_T_body.x, root_T_body.y]) - origin_xy))
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
                    end_reason = "toppled" if mode == "topple" else "contact_lost"
                    break
                if not latch.made and (time.time() - entered) > self.topple_load_grace:
                    end_reason = "contact_lost"
                    break
                if (self.recenter_clearance_gate and clearance is not None
                        and clearance <= 0.0):
                    end_reason = "clearance"
                    break
                if travel > cap:
                    end_reason = "topple_travel" if mode == "topple" else "assist_travel"
                    break
                if mode == "reach":
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
        try:
            self.command_client.robot_command(
                RobotCommandBuilder.synchro_stand_command())
        except Exception:
            rospy.logwarn("ASSIST[%s]: stand failed; base may still be moving.",
                          mode, exc_info=True)

    def _hold_pose(self, goal, root, snap, start_elev, prev_cmd=None, dt=0.0):
        root_T_wr1 = get_a_tform_b(snap, root, WR1_FRAME_NAME)
        tip = np.array(root_T_wr1.transform_point(
            self.tool_tip_x, 0.0, self.tool_tip_z))
        qm = root_T_wr1.rot
        q = self._track_quat(goal, (qm.w, qm.x, qm.y, qm.z), prev_cmd, start_elev, dt)
        return tip, q

    def _await_recontact(self, goal, root, peak, drift, ceiling, start_world,
                         start_elev):
        st = self.state_client.get_robot_state()
        snap = st.kinematic_state.transforms_snapshot
        tip_root, q_cmd = self._hold_pose(goal, root, snap, start_elev)
        load, m = self._contact_force(st)

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
                        waiting, broke_at, loaded_since = True, elapsed, None
                        back = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                               self.contact_lost_grace)
                        self._vlog("ASSIST[topple]: recontact - object broke away at "
                                   "t=%.2fs; holding for it to fall back.", elapsed)
                    elif (latch.made and loaded_since is not None
                          and (now - loaded_since) > self.recontact_settle):
                        reason = "held"
                        break
                else:
                    if back is None:
                        back = el.ContactLatch(self.contact_made_n, self.contact_eps,
                                               self.contact_lost_grace)
                    back.update(load, now)
                    if back.made:
                        tip_root, reason = tip_meas, "recontact"
                        break

                if elapsed > self.recontact_wait:
                    if waiting:
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
                reach_now = el.tip_reach_from_shoulder(snap, self._tip_from_hand)

                now = time.time()
                dt, last_tick = now - last_tick, now
                qm = get_a_tform_b(snap, root, WR1_FRAME_NAME).rot
                q_cmd = self._track_quat(goal, (qm.w, qm.x, qm.y, qm.z), q_cmd,
                                         start_elev, dt)
                aim_err, pivot, tip_live = self._aim_heading_err(goal, snap, root)
                self._publish_aim_viz(root, tip_live, None, pivot,
                                      self._body_x(root_T_body), throttle=True)
                v_rot = self._aim_yaw_rate(aim_err)
                mob = el.build_velocity_mobility_command(
                    self.recenter_body_vel, v_rot=v_rot,
                    max_lin_vel=self.recenter_body_vel,
                    max_ang_vel=max(self.aim_yaw_rate, 1e-3))
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

                if latch.lost:
                    abort = "toppled"
                    break
                if not latch.made and (now - entered) > self.topple_load_grace:
                    abort = "contact_lost"
                    break
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
        q = get_a_tform_b(snap, root, GRAV_ALIGNED_BODY_FRAME_NAME).rot
        R_root_flat = el.quat_wxyz_to_rotmat((q.w, q.x, q.y, q.z))
        return el.rotmat_to_quat_wxyz(
            R_root_flat.T @ el.quat_wxyz_to_rotmat(quat_root_wxyz))

    def _aim_heading_err(self, goal, snap, root):
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
        return "n/a" if aim_err is None else "%+.1fdeg" % np.rad2deg(aim_err)

    @staticmethod
    def _body_x(root_T_body):
        yaw = root_T_body.rot.to_yaw()
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def _recenter_gauge(self, robot_state, snap):
        if self.reach_signal == "manipulability":
            w = el.arm_manipulability(robot_state.kinematic_state.joint_states)
            return ((not np.isnan(w)) and w >= self.recenter_manip_min, w,
                    self.recenter_manip_min)
        reach = el.tip_reach_from_shoulder(snap, self._tip_from_hand)
        return reach <= self.recenter_target_reach, reach, self.recenter_target_reach

    def _clearance(self, goal):
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
        if clearance is None:
            return "n/a"
        if not np.isfinite(clearance):
            return "inf"
        return "%.3fm" % clearance

    def _reach_gauge(self, robot_state, snap):
        if self.reach_signal == "manipulability":
            w = el.arm_manipulability(robot_state.kinematic_state.joint_states)
            return (not np.isnan(w)) and w < self.manip_min, w, self.manip_min
        reach = el.tip_reach_from_shoulder(snap, self._tip_from_hand)
        return reach >= self.max_tip_reach, reach, self.max_tip_reach

    def _coordinated_retreat(self, goal, root, push_dir_world):
        push_dir = np.asarray(push_dir_world, dtype=np.float64)
        if np.linalg.norm(push_dir) < 1e-9:
            return
        push_dir = push_dir / np.linalg.norm(push_dir)
        _, q_hold = el.task_frame_from_force(push_dir)

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

        m0, _ = self._contact_force(self.state_client.get_robot_state())
        state = "RETRACT" if (m0 is None or m0 < self.contact_eps) else "RELEASE"
        self._vlog("RETREAT: start state=%s (F0=%s).", state,
                   "n/a" if m0 is None else "%.1fN" % m0)

        rate = rospy.Rate(self.loop_rate)
        deadline = time.time() + self.retreat_deadline
        release_start = None
        retract_gated = False
        misses = 0
        base_backing = False
        last_base_cmd = 0.0
        clearance = 0.0

        try:
            while not rospy.is_shutdown():
                if time.time() >= deadline:
                    rospy.logwarn("RETREAT: deadline reached with tip-object %.3fm < "
                                  "%.3fm, homing arm.", clearance, self.safe_object_dist)
                    break

                st = self.state_client.get_robot_state()
                snap = st.kinematic_state.transforms_snapshot
                m, raw = self._contact_force(st)
                cur = 0.0 if raw is None else raw
                self.force_pub.publish(Float32(cur))

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

                if not retract_gated:
                    self._gate("contact lost -> retract clear of object?")
                    retract_gated = True

                root_T_hand = get_a_tform_b(snap, root, HAND_FRAME_NAME)
                tip_now = np.array(root_T_hand.transform_point(*self._tip_from_hand))
                clearance = float(np.linalg.norm(tip_now - contact_live))
                if clearance >= self.safe_object_dist:
                    self._vlog("RETREAT: retract done, tip %.3fm >= safe_object_dist "
                               "%.3fm.", clearance, self.safe_object_dist)
                    break

                target = contact_live - self.retreat_dist * push_dir
                target, clamp_binds = el.clamp_target_to_body(
                    snap, root, target, self.min_hand_body_x)

                now = time.time()
                mob, mob_is_se2 = None, False
                if clamp_binds:
                    body_T_hand = get_a_tform_b(snap, GRAV_ALIGNED_BODY_FRAME_NAME,
                                                HAND_FRAME_NAME)
                    tip_x = body_T_hand.transform_point(*self._tip_from_hand)[0]
                    base_xy = el.base_align_pose(contact_live, push_dir,
                                                 tip_x + self.safe_object_dist)
                    if base_xy is not None:
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
                    mob = stand_mob
                    base_backing = False
                    self._vlog("RETREAT: clamp released, base re-planted.")

                cmd, _ = el.build_standoff_pose_command(
                    target, push_dir, q_hold, root_frame=root, eps=0.0,
                    duration_s=self.retract_move_time, tool_offset=self._tool_offset,
                    mobility_command=mob)
                if mob_is_se2:
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
        carry_cmd = el.build_carry_pose_command(
            self.carry_hand_x, self.carry_hand_z,
            root_frame=GRAV_ALIGNED_BODY_FRAME_NAME,
            duration_s=self.carry_move_time,
            max_lin_vel=self.carry_max_lin_vel, max_accel=self.carry_max_accel,
            mobility_command=el.build_stand_mobility_command())
        carry_id = self.command_client.robot_command(carry_cmd)
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
        if self.verbose:
            rospy.loginfo(fmt, *args)

    def _gate(self, msg):
        if not self.interactive:
            return
        try:
            input("[step] %s -- press Enter to continue... " % msg)
        except EOFError:
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
        now = time.time()
        if self.clr_viz_rate <= 0.0 or (now - self._clr_viz_last) < (1.0 / self.clr_viz_rate):
            return
        self._clr_viz_last = now
        frame = GRAV_ALIGNED_BODY_FRAME_NAME
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
        m.scale.z = 0.1
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
        m.scale.x = 0.01
        m.scale.y = 0.02
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

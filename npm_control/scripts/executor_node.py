#!/usr/bin/env python3
"""executor_node; push executor as an ExecutePush action server.
One macro-step push = one long-running, preemptable action goal.
The node works in two phases: a) align, where the arm aligns with the force direction,
and b) approach and push, where the object is pushed at loop rate of 30 to 50 Hz.
For testing purposes, the node can run be run using python arguments which overrides 
the ROS private params, e.g. `_exec/dry_run:=true _viz/arrow_len:=0.5`.

Run example:
    # 1) Make sure an estop script is running in another terminal (estop SDK example), then:
    rosrun npm_control executor_node_real.py _exec/dry_run:=false _exec/hostname:=ROBOT_IP
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
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from npm_msgs.msg import (ExecutePushAction, ExecutePushFeedback, ExecutePushResult)

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.frame_helpers import (ODOM_FRAME_NAME, VISION_FRAME_NAME,
                                         HAND_FRAME_NAME, get_a_tform_b)
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

# Add the package src so this runs even without the workspace devel sourced.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from npm_control import executor_lib as el


def _ee_force_mag(state_client):
    # Magnitude (N) of the estimated end-effector force, or None if unavailable.
    st = state_client.get_robot_state()
    man = st.manipulator_state
    if not man.HasField("estimated_end_effector_force_in_hand"):
        return None
    f = man.estimated_end_effector_force_in_hand
    return float(np.linalg.norm([f.x, f.y, f.z]))


class ExecutorNodeReal(object):
    def __init__(self):
        rf = rospy.get_param("~exec/root_frame", "odom")
        if rf not in ("odom", "vision"):
            raise ValueError("~exec/root_frame must be 'odom' or 'vision', got %r" % rf)
        self.root_frame = {"odom": ODOM_FRAME_NAME, "vision": VISION_FRAME_NAME}[rf]
        self.object_frame = rospy.get_param("~exec/object_frame", "object_1")
        self.dry_run = rospy.get_param("~exec/dry_run", True)
        self.force_clip = rospy.get_param("~exec/force_clip", el.FORCE_NORM_CLIP)
        self.watchdog_margin = rospy.get_param("~exec/watchdog_margin", 15.0)
        self.arrow_len = rospy.get_param("~viz/arrow_len", 1.0)
        self.approach_eps = rospy.get_param("~exec/approach_eps", 0.10)
        self.loop_rate = rospy.get_param("~exec/loop_rate", 40.0)
        self.cmd_horizon = rospy.get_param("~exec/cmd_horizon", 0.6)
        self.contact_eps = rospy.get_param("~exec/contact_eps", 2.0)
        self.approach_force = rospy.get_param("~exec/approach_force", 6.0)
        self.contact_made_n = rospy.get_param("~exec/contact_made_n", 5.0)
        # env-level (npm.yaml), flat key ~max_finger_reach, not ~exec/
        self.max_finger_reach = rospy.get_param("~max_finger_reach", 0.6)
        self.ramp_time = rospy.get_param("~exec/ramp_time", 0.5)
        self.tf_timeout = rospy.get_param("~exec/tf_timeout", 0.2)
        self.retreat_dist = rospy.get_param("~exec/retreat_dist", 0.2)

        self.robot = None
        self.command_client = None
        self.state_client = None
        self.lease_client = None
        if not self.dry_run:
            self._connect()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.marker_pub = rospy.Publisher("/npm/push_marker", Marker, queue_size=1)
        # Start action server after power on sequence.
        # auto_start=False so we call start() at the end of run().
        self.server = actionlib.SimpleActionServer(
            "push", ExecutePushAction, execute_cb=self._execute_cb, auto_start=False)
        rospy.loginfo("executor_node up (dry_run=%s, root_frame=%s, object_frame=%s)",
                      self.dry_run, self.root_frame, self.object_frame)

    def _connect(self):
        verbose = rospy.get_param("~exec/verbose", False)
        hostname = rospy.get_param("~exec/hostname", None)
        if not hostname:
            raise ValueError("~exec/hostname is required when ~exec/dry_run is false")
        bosdyn.client.util.setup_logging(verbose)
        sdk = bosdyn.client.create_standard_sdk("NpmExecutorRealClient")
        self.robot = sdk.create_robot(hostname)
        bosdyn.client.util.authenticate(self.robot) 
        self.robot.time_sync.wait_for_sync()

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

            self.robot.logger.info("Unstowing arm...")
            unstow_id = self.command_client.robot_command(
                RobotCommandBuilder.arm_ready_command())
            block_until_arm_arrives(self.command_client, unstow_id, 3.0)

            rospy.on_shutdown(self._safe_stop)
            # Starting the action server after the robot is powered on, standing, and unstowed.
            self.server.start()  
            self.robot.logger.info("Ready - ExecutePush action server up.")
            try:
                rospy.spin()
            finally:
                self._stow_and_power_off()

    def _object_pose_in_root(self, object_frame):
        # Look up the tracked object's pose in the root frame.
        try:
            tf = self.tf_buffer.lookup_transform(self.root_frame, object_frame,
                                                 rospy.Time(0),
                                                 rospy.Duration(self.tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "TF %s<-%s lookup failed: %s",
                                   self.root_frame, object_frame, exc)
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return R, np.array([t.x, t.y, t.z])

    def _track(self, goal):
        """World contact + force for the goal, from the current object pose.

        The object TF frame comes from the goal; empty -> the executor's default
        (--object-frame), mirroring root_frame.
        """
        object_frame = goal.object_frame or self.object_frame
        pose = self._object_pose_in_root(object_frame)
        if pose is None:
            return None
        R, t = pose
        push_point = [goal.push_point.x, goal.push_point.y, goal.push_point.z]
        force_body = [goal.force_body.x, goal.force_body.y, goal.force_body.z]
        return el.world_contact_from_object(R, t, push_point, force_body)

    def _execute_cb(self, goal):
        root = {"": self.root_frame, "odom": ODOM_FRAME_NAME,
                "vision": VISION_FRAME_NAME}.get(goal.root_frame, self.root_frame)
        track = self._track(goal)
        if track is None:
            self._abort("error", "object TF unavailable at goal start")
            return
        contact_world, force_world = track
        mag, q = el.task_frame_from_force(force_world)
        push_dir = force_world / max(np.linalg.norm(force_world), 1e-9)
        rospy.loginfo("GOAL loc=%d |F|=%.1fN root=%s contact=(%.3f,%.3f,%.3f) dur=%.1fs%s",
                      goal.loc_idx, mag, root, contact_world[0], contact_world[1],
                      contact_world[2], goal.duration,
                      " [dry_run]" if self.dry_run else "")
        self._publish_marker(root, contact_world, force_world, mag)

        if self.dry_run:
            res = ExecutePushResult(contact_made=False, peak_force=0.0,
                                    finger_travel=0.0, end_reason="dry_run")
            self.server.set_succeeded(res)
            return

        # align phase. free-space move to the standoff behind the contact.
        self._publish_feedback(ExecutePushFeedback.PHASE_ALIGN, False, 0.0, 0.0)
        standoff_cmd, _ = el.build_standoff_pose_command(
            contact_world, push_dir, q, root_frame=root, eps=self.approach_eps)
        try:
            sid = self.command_client.robot_command(standoff_cmd)
            block_until_arm_arrives(self.command_client, sid, 5.0)
        except Exception:
            rospy.logerr("ALIGN move failed - stopping arm, aborting goal.", exc_info=True)
            self._stop_arm()
            self._abort("error", "align failed")
            return

        self._push_loop(goal, root)

    def _push_loop(self, goal, root):
        """
        Approach+Push phase: continuously issue the hybrid force command while tracking the object.
        """
        # Push-point start = world contact at the moment the force drive begins.
        start = self._track(goal)
        start_world = start[0] if start is not None else None
        peak = 0.0
        contact_made = False
        contact_time = None  # wall time contact first detected;
        ceiling = self.force_clip + self.watchdog_margin
        rate = rospy.Rate(self.loop_rate)
        deadline = time.time() + float(goal.duration)
        end_reason, drift = "timeout", 0.0
        last_force_world = None

        try:
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    end_reason = "preempted"
                    break

                track = self._track(goal)
                if track is not None:
                    contact_world, force_world = track
                    last_force_world = force_world
                    if start_world is None:
                        start_world = contact_world
                    drift = float(np.linalg.norm(contact_world - start_world))
                    # Force ramp, instead of suddenly applying full force.
                    full_mag = float(np.linalg.norm(force_world))
                    if contact_time is None:
                        desired_mag = min(self.approach_force, full_mag)
                    elif self.ramp_time > 0.0:
                        frac = min(1.0, (time.time() - contact_time) / self.ramp_time)
                        desired_mag = self.approach_force + frac * (full_mag - self.approach_force)
                    else:
                        desired_mag = full_mag
                    force_cmd = force_world * (desired_mag / max(full_mag, 1e-9))
                    # Reissue since the latest command supersedes the running trajectory at once,
                    # so force is continuous; cmd_horizon > loop period -> never expires.
                    cmd, _, _ = el.build_arm_cartesian_command(
                        contact_world, force_cmd, root_frame=root,
                        duration_s=self.cmd_horizon, force_clip=self.force_clip,
                        follow_arm=True)
                    self.command_client.robot_command(cmd)

                m = _ee_force_mag(self.state_client)
                cur = 0.0 if m is None else m
                if m is not None:
                    peak = max(peak, m)
                    if m > self.contact_made_n:
                        if not contact_made:
                            contact_time = time.time() 
                        contact_made = True
                    if m > ceiling:
                        rospy.logerr("WATCHDOG: EE force %.1fN > %.1fN - stopping arm.",
                                     m, ceiling)
                        self._stop_arm()
                        end_reason = "watchdog"
                        break

                self._publish_feedback(
                    ExecutePushFeedback.PHASE_PUSH if contact_made
                    else ExecutePushFeedback.PHASE_APPROACH, contact_made, cur, drift)

                # SMDP stops when push point is out of reach or contact is lost.
                if drift > self.max_finger_reach:
                    end_reason = "reach"
                    break
                if contact_made and m is not None and m < self.contact_eps:
                    end_reason = "contact_lost"
                    break
                if time.time() >= deadline:
                    end_reason = "timeout"
                    break
                rate.sleep()
        except Exception:
            # for safety, stop the arm, do not retry. Bring the node down safely.
            rospy.logerr("Exception during push - stopping arm, no retry.", exc_info=True)
            self._stop_arm()
            self._abort("error", "push exception", peak=peak,
                        contact_made=contact_made, drift=drift)
            rospy.signal_shutdown("push exception")
            return

        # Marker at the final tracked pose corresponding to the best-effort.
        if last_force_world is not None:
            track = self._track(goal)
            if track is not None:
                self._publish_marker(root, track[0], last_force_world,
                                     float(min(np.linalg.norm(last_force_world),
                                               self.force_clip)))

        res = ExecutePushResult(contact_made=contact_made, peak_force=peak,
                                finger_travel=drift, end_reason=end_reason)
        rospy.loginfo("PushResult: contact=%s peak=%.1fN drift=%.3fm reason=%s",
                      contact_made, peak, drift, end_reason)
        if end_reason == "preempted":
            self.server.set_preempted(res)
        else:
            self.server.set_succeeded(res)
        self._return_to_ready(last_force_world)

    def _retreat_hand(self, retreat_dir):
        """
        Pull the hand straight back off the surface along -retreat_dir (position move).
        """
        d = np.asarray(retreat_dir, dtype=np.float64)
        n = np.linalg.norm(d)
        if n < 1e-9 or self.retreat_dist <= 0.0:
            return True 
        snap = self.state_client.get_robot_state().kinematic_state.transforms_snapshot
        root_T_hand = get_a_tform_b(snap, self.root_frame, HAND_FRAME_NAME)
        hand_pos = [root_T_hand.x, root_T_hand.y, root_T_hand.z]
        q = root_T_hand.rot  # wxyz
        cmd, _ = el.build_standoff_pose_command(
            hand_pos, d / n, [q.w, q.x, q.y, q.z], root_frame=self.root_frame,
            eps=self.retreat_dist, duration_s=1.5, follow_arm=True)
        rid = self.command_client.robot_command(cmd)
        return block_until_arm_arrives(self.command_client, rid, 3.0)

    def _return_to_ready(self, retreat_dir=None):
        """
        Home the arm after a push: retreat the hand off the surface (body follows to make
        room), then drive to the ready/unstow pose.
        """
        try:
            self._retreat_hand(retreat_dir)
            ready_id = self.command_client.robot_command(
                RobotCommandBuilder.arm_ready_command())
            block_until_arm_arrives(self.command_client, ready_id, 3.0)
            rospy.loginfo("Arm returned to ready.")
        except Exception:
            rospy.logerr("Return-to-ready failed.", exc_info=True)

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

    def _publish_feedback(self, phase, contact_made, cur_force, point_drift):
        fb = ExecutePushFeedback()
        fb.phase = phase
        fb.contact_made = bool(contact_made)
        fb.cur_force = float(cur_force)
        fb.point_drift = float(point_drift)
        self.server.publish_feedback(fb)

    def _publish_marker(self, frame, contact, force, fmag):
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

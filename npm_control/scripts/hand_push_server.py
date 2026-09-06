#!/usr/bin/env python3
"""
ExecutePush action server with a human as the actuator: it shows where to push
and waits for you, so the coordinator's macro-step loop can be tested with no
robot. Never commands the robot, so no lease and no estop are needed.

Example:
    rosrun npm_control hand_push_server.py _npz_path:=/path/to/object.npz
"""
import os
import sys

import numpy as np
import rospy
import actionlib
import tf2_ros
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker
from npm_msgs.msg import (ExecutePushAction, ExecutePushFeedback, ExecutePushResult)

# Same path fix-up as executor_node: run from the source tree without devel.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "npm_policy", "src"))
from npm_control import executor_lib as el
from npm_policy import obs_lib as ol


class HandPushServer(object):
    def __init__(self):
        self.npz_path = rospy.get_param("~npz_path", "")
        # Contact-cloud margin; must match policy_node's, and _model_cb checks it.
        self.pcl_shrink = float(rospy.get_param("~pcl_shrink", 1.0))
        # world, not odom: there is no Spot driver in this mode, so odom does not
        # exist. Goals arrive with root_frame empty and inherit this.
        self.root_frame = rospy.get_param("~root_frame", "world")
        self.object_frame = rospy.get_param("~object_frame", "object_link")
        self.move_thr = float(rospy.get_param("~hand/move_thr", 0.02))
        self.arrow_len = float(rospy.get_param("~viz/arrow_len", 0.3))
        self.tf_timeout = float(rospy.get_param("~exec/tf_timeout", 0.5))

        self.link_pts = None
        if self.npz_path:
            pts, _n, scale, centroid = ol.load_model_npz(self.npz_path)
            self.link_pts = ol.link_frame_cloud(pts, scale, centroid,
                                                self.pcl_shrink)
            rospy.loginfo("object model: %s (%d points, scale=%.4f pcl_shrink=%.4f)",
                          self.npz_path, self.link_pts.shape[0], scale,
                          self.pcl_shrink)
        else:
            rospy.logwarn("~npz_path unset: loc_idx goals will be REJECTED.")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.marker_pub = rospy.Publisher("/npm/push_marker", Marker, queue_size=3)
        self.model_checked = False
        rospy.Subscriber("/npm/model_info", String, self._model_cb, queue_size=1)
        # ... but only if something publishes it. Nothing does when the stack runs
        # without policy_node, and an unverified model is silent otherwise.
        self.model_timeout = float(rospy.get_param("~model_info_timeout", 5.0))
        rospy.Timer(rospy.Duration(self.model_timeout), self._model_timeout,
                    oneshot=True)

        self.server = actionlib.SimpleActionServer(
            "push", ExecutePushAction, execute_cb=self._execute_cb, auto_start=False)
        self.server.start()
        rospy.loginfo("hand_push_server up (root=%s object=%s). Push by hand; the "
                      "server waits for Enter.", self.root_frame, self.object_frame)

    def _model_cb(self, msg):
        # loc_idx is only an index: if the policy and this node hold different
        # models, the same index is a different point and the wrong face is pushed.
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

    def _object_pose(self, object_frame):
        try:
            tf = self.tf_buffer.lookup_transform(self.root_frame, object_frame,
                                                 rospy.Time(0),
                                                 rospy.Duration(self.tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "TF %s<-%s lookup failed: %s",
                                   self.root_frame, object_frame, exc)
            return None
        q, t = tf.transform.rotation, tf.transform.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return R, np.array([t.x, t.y, t.z])

    def _execute_cb(self, goal):
        object_frame = goal.object_frame or self.object_frame
        try:
            push_point, src = el.resolve_push_point(goal.loc_idx, goal.push_point,
                                                    self.link_pts)
        except ValueError as exc:
            self._abort(str(exc))
            return

        pose = self._object_pose(object_frame)
        if pose is None:
            self._abort("object TF unavailable at goal start")
            return
        R, t_start = pose
        force_body = [goal.force_body.x, goal.force_body.y, goal.force_body.z]
        contact, force = el.world_contact_from_object(R, t_start, push_point,
                                                      force_body)
        mag = float(np.linalg.norm(force))
        rospy.loginfo("GOAL loc=%d src=%s |F|=%.1fN root=%s point=(%.3f,%.3f,%.3f) "
                      "contact=(%.3f,%.3f,%.3f)", goal.loc_idx, src, mag,
                      self.root_frame, push_point[0], push_point[1], push_point[2],
                      contact[0], contact[1], contact[2])
        self._publish_markers(contact, force, mag, goal.loc_idx)
        self._publish_feedback()

        end_reason = self._wait_for_hand(goal, contact, force, mag)

        pose = self._object_pose(object_frame)
        travel = 0.0 if pose is None else float(np.linalg.norm(pose[1] - t_start))
        res = ExecutePushResult(contact_made=travel > self.move_thr,
                                peak_force=0.0, finger_travel=travel,
                                end_reason=end_reason)
        rospy.loginfo("hand push done: reason=%s object moved %.3fm", end_reason,
                      travel)
        self.server.set_succeeded(res)

    def _wait_for_hand(self, goal, contact, force, mag):
        # Blocking on stdin is what makes the rig operator-paced; roslaunch gives
        # a node no tty, so fall back to a timed wait and say so.
        direction = force / max(mag, 1e-9)
        print("\n" + "=" * 68)
        print(" PUSH BY HAND    loc_idx = %d" % goal.loc_idx)
        print("   at %s (%.3f, %.3f, %.3f)"
              % (self.root_frame, contact[0], contact[1], contact[2]))
        print("   along (%.2f, %.2f, %.2f)   |F| = %.1f N"
              % (direction[0], direction[1], direction[2], mag))
        print(" See the green arrow in rviz. Press ENTER when the object has "
              "stopped.")
        print("=" * 68)
        sys.stdout.flush()
        try:
            input()
            return "hand"
        except (EOFError, OSError):
            rospy.logwarn("no stdin (not in an xterm?): waiting %.1fs instead of "
                          "for Enter.", goal.duration)
            rospy.sleep(max(goal.duration, 0.1))
            return "eof_timeout"

    def _abort(self, msg):
        rospy.logwarn("Goal aborted: %s", msg)
        self.server.set_aborted(ExecutePushResult(end_reason="error"), msg)

    def _publish_feedback(self):
        fb = ExecutePushFeedback()
        fb.phase = ExecutePushFeedback.PHASE_PUSH
        fb.contact_made = False
        fb.cur_force = 0.0
        fb.point_drift = 0.0
        self.server.publish_feedback(fb)

    def _publish_markers(self, contact, force, mag, loc_idx):
        # Same topic, namespace and ids as executor_node._publish_marker, so one
        # rviz display serves both rigs. Only one of the two ever runs.
        stamp = rospy.Time.now()
        scale = (self.arrow_len * min(mag, el.FORCE_NORM_CLIP) / el.FORCE_NORM_CLIP) \
            / max(mag, 1e-6)

        arrow = self._marker(0, Marker.ARROW, stamp)
        arrow.points = [Point(*[float(v) for v in contact]),
                        Point(*[float(contact[i] + scale * force[i])
                                for i in range(3)])]
        arrow.scale.x, arrow.scale.y = 0.01, 0.02
        arrow.color = ColorRGBA(0.1, 1.0, 0.1, 1.0)

        sphere = self._marker(1, Marker.SPHERE, stamp)
        sphere.pose.position = Point(*[float(v) for v in contact])
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.04
        sphere.color = ColorRGBA(1.0, 0.6, 0.1, 0.9)

        text = self._marker(3, Marker.TEXT_VIEW_FACING, stamp)
        text.pose.position = Point(float(contact[0]), float(contact[1]),
                                   float(contact[2]) + 0.12)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.06
        text.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        text.text = "loc=%d  |F|=%.0fN" % (loc_idx, mag)

        for m in (arrow, sphere, text):
            self.marker_pub.publish(m)

    def _marker(self, mid, mtype, stamp):
        m = Marker()
        m.header.frame_id = self.root_frame
        m.header.stamp = stamp
        m.ns = "npm_push"
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.lifetime = rospy.Duration(0.0)
        return m


def main():
    rospy.init_node("hand_push_server")
    HandPushServer()
    rospy.spin()


if __name__ == "__main__":
    main()

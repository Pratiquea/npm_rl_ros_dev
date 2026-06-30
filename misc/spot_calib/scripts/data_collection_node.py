#!/usr/bin/env python3
"""Data collection node: record paired body poses to a rosbag while showing a
live readiness meter that tells the operator when enough non-degenerate motion
has been captured.

Data collection method:walk/change body pose of the robot until the indicator mentions ready, then Ctrl-C.

rosrun spot_calib data_collection_node.py _bag:=/media/ssd/rosbags/npm/spot_calib.bag
"""
import threading

import numpy as np
import rospy
import rosbag
import tf2_ros
import message_filters as mf
from rospy import AnyMsg
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String

from spot_calib import calib_lib as cl


# Extra topics to be recorded
DEFAULT_EXTRA_TOPICS = [
    "/joint_states",
    "/mocap_node/optitrack_config/parameter_descriptions",
    "/mocap_node/optitrack_config/parameter_updates",
    "/mocap_node/spot_body/Odom",
    "/mocap_node/spot_body/ground_pose",
    "/mocap_node/spot_body/pose",
    "/mocap_node/spot_ee_tip/Odom",
    "/mocap_node/spot_ee_tip/ground_pose",
    "/mocap_node/spot_ee_tip/pose",
    "/spot/body_pose/status",
    "/spot/motion_or_idle_body_pose/status",
    "/spot/navigate_route/status",
    "/spot/navigate_to/status",
    "/spot/odometry",
    "/spot/odometry/twist",
    "/spot/odometry_corrected",
    "/spot/status/battery_states",
    "/spot/status/behavior_faults",
    "/spot/status/estop",
    "/spot/status/feedback",
    "/spot/status/feet",
    "/spot/status/leases",
    "/spot/status/metrics",
    "/spot/status/mobility_params",
    "/spot/status/motion_allowed",
    "/spot/status/power_state",
    "/spot/status/system_faults",
    "/spot/status/wifi",
    "/spot/trajectory/status",
    "/spot/world_objects",
    "/tf",
    "/tf_static",
    "/twist_marker_server/update",
    "/twist_marker_server/update_full",
]


def _bar(frac, width=24):
    frac = max(0.0, min(1.0, float(frac)))
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + "] {:3.0f}%".format(frac * 100)


def _tf_to_se3(tf):
    t = tf.transform.translation
    r = tf.transform.rotation
    R = cl.quat_wxyz_to_rotmat([r.w, r.x, r.y, r.z])
    return cl.make_se3(R, [t.x, t.y, t.z])


class CollectNode(object):
    def __init__(self):
        self.world = rospy.get_param("~world_frame", "world")
        self.mocap_body = rospy.get_param("~mocap_body_frame", "spot_body_mocap")
        self.vision = rospy.get_param("~internal_root_frame", "vision")
        self.body = rospy.get_param("~internal_body_frame", "base_link")
        self.period = float(rospy.get_param("~sample_period_s", 0.5))  
        self.bag_path = rospy.get_param("~bag", "/tmp/spot_calib.bag")
        self.n_target = int(rospy.get_param("~n_target", 12))
        self.min_rot_deg = float(rospy.get_param("~min_rot_deg", 5.0))
        self.l2_min = float(rospy.get_param("~l2_min", 0.15))
        self.mocap_topic = rospy.get_param("~mocap_topic",
                                           "/mocap_node/spot_body/pose")
        self.spot_topic = rospy.get_param("~spot_topic", "/spot/odometry")
        self.sync_slop = float(rospy.get_param("~sync_slop_s", 0.02))
        # throttle accepted pairs to ~period
        self._last_kept = rospy.Time(0)    

        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf)
        self.A_abs = []
        self.bag = rosbag.Bag(self.bag_path, "w")
        self.bag_lock = threading.Lock()
        self.closed = False
        self.pub_ready = rospy.Publisher("/calib/readiness", Float32, queue_size=1)
        self.pub_status = rospy.Publisher("/calib/status", String, queue_size=1)
        rospy.on_shutdown(self._close)

        # Record extra topics.
        self.extra_topics = [t for t in
                             rospy.get_param("~extra_topics", DEFAULT_EXTRA_TOPICS)
                             if t not in (self.mocap_topic, self.spot_topic)]
        self.extra_subs = [
            rospy.Subscriber(t, AnyMsg, self._record_any, callback_args=t,
                             queue_size=50)
            for t in self.extra_topics
        ]

        rospy.loginfo("collect_node: writing %s. Move the robot through varied "
                      "rotations (yaw + pitch/roll). Ctrl-C when READY.",
                      self.bag_path)
        rospy.loginfo("collect_node: also recording %d extra topics.",
                      len(self.extra_topics))

        self.a_sub = mf.Subscriber(self.mocap_topic, PoseStamped)
        self.b_sub = mf.Subscriber(self.spot_topic, Odometry)
        self.sync = mf.ApproximateTimeSynchronizer(
            [self.a_sub, self.b_sub], queue_size=50, slop=self.sync_slop)
        self.sync.registerCallback(self._on_pair)
        rospy.loginfo("collect_node: sync %s + %s (slop=%.0f ms)",
                      self.mocap_topic, self.spot_topic, self.sync_slop * 1e3)

    def _record_any(self, msg, topic):
        """Write any topic to the bag without knowing its type"""
        try:
            with self.bag_lock:
                if self.closed:
                    return
                self.bag.write(topic, msg, rospy.Time.now(),
                               connection_header=msg._connection_header)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "collect_node: record %s failed: %s",
                                   topic, e)

    def _lookup(self, parent, child, stamp):
        return self.buf.lookup_transform(parent, child, stamp,
                                         rospy.Duration(0.2))

    def _on_pair(self, a_msg, b_msg):
        """Synced mocap+spot messages."""
        ta = a_msg.header.stamp
        tb = b_msg.header.stamp
        # throttle
        if (ta - self._last_kept).to_sec() < self.period:
            return
        try:
            A_tf = self._lookup(self.world, self.mocap_body, ta)
            B_tf = self._lookup(self.vision, self.body, tb)
        except Exception as e:  
            rospy.logwarn_throttle(2.0, "collect_node: TF not ready: %s", e)
            return
        self._last_kept = ta

        A_tf.header.stamp = ta
        B_tf.header.stamp = tb
        with self.bag_lock:
            if self.closed:
                return
            self.bag.write("/calib/A", A_tf, ta)
            self.bag.write("/calib/B", B_tf, tb)

        self.A_abs.append(_tf_to_se3(A_tf))
        m = cl.excitation(self.A_abs, self.min_rot_deg, self.n_target, self.l2_min)

        # readiness, check for sample coverage and axis-spread coverage
        spread_frac = min(1.0, m["l2r"] / self.l2_min) if self.l2_min > 0 else 1.0
        readiness = min(m["sample_frac"], spread_frac)
        self.pub_ready.publish(Float32(readiness))
        flag = "READY  - Ctrl-C to stop" if m["ready"] else "collecting..."
        status = ("rot-samples %s  axis-spread %s  (l2r=%.2f spread=%.0fdeg)  %s"
                  % (_bar(m["sample_frac"]), _bar(spread_frac),
                     m["l2r"], m["spread_deg"], flag))
        self.pub_status.publish(String(status))
        rospy.loginfo(status)

    def _close(self):

        for s in (getattr(self, "a_sub", None), getattr(self, "b_sub", None)):
            try:
                s.sub.unregister()
            except Exception:
                pass
        for s in getattr(self, "extra_subs", []):
            try:
                s.unregister()
            except Exception:
                pass
        try:
            with self.bag_lock:
                self.closed = True
                self.bag.close()
            rospy.loginfo("collect_node: closed %s (%d samples). Now run "
                          "calibrate_from_bag.py.", self.bag_path, len(self.A_abs))
        except Exception:
            pass


def main():
    rospy.init_node("collect_node")
    CollectNode()
    rospy.spin()


if __name__ == "__main__":
    main()
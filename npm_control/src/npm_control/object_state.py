#!/usr/bin/env python3
"""
Filtered object state in `world` from the mocap pose stream: constant-velocity
Kalman filters for linear and angular velocity, plus distance/direction to the
goal (the world origin).

ObjectStateEstimator is the publish side, run by exactly one process
(object_state_node). ObjectStateClient is the subscribe side every consumer uses.
"""
import threading
from collections import namedtuple

import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs                                   # noqa: F401 - do_transform_pose
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from npm_msgs.msg import ObjectState

from npm_control.kalman import ConstantVelocityKF, AngularRateEstimator

Snapshot = namedtuple("Snapshot", "stamp pos rot lin_vel ang_vel")


class ObjectStateEstimator(object):
    """Subscribes the mocap object pose, publishes the filtered state.

    Poses arrive in `mocap_world`; the goal-centered `world` frame is one static
    rotation away, so the transform is looked up once and reused. The mocap
    rigid body's pivot is not the model's link origin, so the calibrated
    `<mocap body> -> object_link` extrinsic (npm_calib) is applied on top. Every quantity
    exposed is in `world` and refers to the object's LINK origin, which is what
    obs_lib.link_frame_cloud and policy_lib.to_world assume.

    Run by object_state_node only: one instance in the system, so every consumer
    sees the same filter state rather than its own copy.
    """

    def __init__(self):
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.object_frame = rospy.get_param("~object_frame", "object_link")
        self.pose_topic = rospy.get_param("~object_pose_topic", "/object/pose")
        self.lookup_timeout = float(rospy.get_param("~lookup_timeout", 0.2))
        self.max_gap = float(rospy.get_param("~kf_max_gap_s", 0.2))
        self.publish_state = bool(rospy.get_param("~publish_state", True))
        self.mocap_frame = rospy.get_param("~object_parent_frame", "object_1")
        self.link_frame = rospy.get_param("~object_extrinsic_frame", "object_link")

        self.lin_kf = ConstantVelocityKF(rospy.get_param("~kf_sigma_a", 1.0),
                                         rospy.get_param("~kf_sigma_meas", 0.001))
        self.ang_est = AngularRateEstimator(
            rospy.get_param("~kf_sigma_alpha", 1.0),
            rospy.get_param("~kf_sigma_meas_ang", 0.001))

        self.lock = threading.Lock()
        self.snap = None
        self.tf_world_mocap = None
        self.extrinsic = None                       # (Rotation, t) mocap body -> object_link
        self.prev_t = None

        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf)
        self.state_pub = rospy.Publisher("/npm/object_state", ObjectState,
                                         queue_size=1)
        self.odom_pub = rospy.Publisher("/npm/object_odom", Odometry, queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.loginfo("object_state: %s -> %s (sigma_a=%.3g sigma_meas=%.3g)",
                      self.pose_topic, self.world_frame, self.lin_kf.sigma_a,
                      self.lin_kf.sigma_meas)

    def ready(self):
        return self.snap is not None

    def snapshot(self):
        with self.lock:
            return self.snap

    def dist_dir(self):
        """(dist, unit xy direction object -> goal). Goal is the world origin."""
        snap = self.snapshot()
        if snap is None:
            return float("nan"), np.zeros(2)
        xy = snap.pos[:2]
        dist = float(np.linalg.norm(xy))
        return dist, (-xy / max(dist, 1e-6))

    def to_msg(self):
        snap = self.snapshot()
        if snap is None:
            return None
        dist, direction = self.dist_dir()
        x, y, z, w = snap.rot.as_quat()

        m = ObjectState()
        m.header.stamp = snap.stamp
        m.header.frame_id = self.world_frame
        m.pose.position.x = float(snap.pos[0])
        m.pose.position.y = float(snap.pos[1])
        m.pose.position.z = float(snap.pos[2])
        m.pose.orientation.x = float(x)
        m.pose.orientation.y = float(y)
        m.pose.orientation.z = float(z)
        m.pose.orientation.w = float(w)
        m.lin_vel.x = float(snap.lin_vel[0])
        m.lin_vel.y = float(snap.lin_vel[1])
        m.lin_vel.z = float(snap.lin_vel[2])
        m.ang_vel.x = float(snap.ang_vel[0])
        m.ang_vel.y = float(snap.ang_vel[1])
        m.ang_vel.z = float(snap.ang_vel[2])
        m.dist_to_goal = dist
        m.dir_to_goal_x, m.dir_to_goal_y = float(direction[0]), float(direction[1])
        return m

    def _to_world(self, msg):
        if self.tf_world_mocap is None:
            try:
                self.tf_world_mocap = self.buf.lookup_transform(
                    self.world_frame, msg.header.frame_id, rospy.Time(0),
                    rospy.Duration(self.lookup_timeout))
            except tf2_ros.TransformException as e:
                rospy.logwarn_throttle(2.0, "object_state: TF %s <- %s: %s",
                                       self.world_frame, msg.header.frame_id, e)
                return None
        return tf2_geometry_msgs.do_transform_pose(msg, self.tf_world_mocap)

    def _apply_extrinsic(self, pos, rot):
        """Compose the mocap pivot pose with the extrinsic to get the link pose.

        This is a RIGHT-multiply (`T_world_link = T_world_body . body_T_link`), so it
        cannot go through do_transform_pose the way the world hop does. Looked up
        once and cached, like tf_world_mocap; a missing extrinsic falls back to
        identity so an uncalibrated stack still streams state.
        """
        if self.extrinsic is None:
            try:
                tf = self.buf.lookup_transform(self.mocap_frame, self.link_frame,
                                               rospy.Time(0),
                                               rospy.Duration(self.lookup_timeout))
            except tf2_ros.TransformException as e:
                rospy.logwarn_throttle(5.0, "object_state: no extrinsic %s <- %s "
                                       "(%s); using identity", self.mocap_frame,
                                       self.link_frame, e)
                return pos, rot
            t = tf.transform.translation
            q = tf.transform.rotation
            self.extrinsic = (Rotation.from_quat([q.x, q.y, q.z, q.w]),
                              np.array([t.x, t.y, t.z]))
            rospy.loginfo("object_state: extrinsic %s <- %s t=%s, r=%s",
                          self.mocap_frame, self.link_frame, self.extrinsic[1], self.extrinsic[0].as_euler("xyz", degrees=True))

        rot_ext, t_ext = self.extrinsic
        return pos + rot.apply(t_ext), rot * rot_ext

    def _pose_cb(self, msg):
        world_pose = self._to_world(msg)
        if world_pose is None:
            return

        t = msg.header.stamp.to_sec()
        if self.prev_t is not None and t <= self.prev_t:
            return                                  # duplicate or out-of-order sample

        p = world_pose.pose.position
        pos_meas = np.array([p.x, p.y, p.z])
        q = world_pose.pose.orientation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        # Before the filters, not after: the extrinsic's lever arm turns object
        # rotation into link translation, and a KF fed the pivot pose would miss
        # that motion entirely.
        pos_meas, rot = self._apply_extrinsic(pos_meas, rot)

        # A dropout longer than max_gap makes the constant-velocity prior wrong;
        # restarting beats emitting a velocity spike into the settle check.
        dt = 0.0 if self.prev_t is None else t - self.prev_t
        if dt <= 0.0 or dt > self.max_gap:
            self.lin_kf.reset(pos_meas)
            self.ang_est.reset(rot)
            pos, lin_vel, ang_vel = pos_meas, np.zeros(3), np.zeros(3)
        else:
            pos, lin_vel = self.lin_kf.update(pos_meas, dt)
            ang_vel = self.ang_est.update(rot, dt)

        self.prev_t = t
        with self.lock:
            self.snap = Snapshot(msg.header.stamp, pos, rot, lin_vel, ang_vel)
        if self.publish_state:
            self._publish()

    def _publish(self):
        state = self.to_msg()
        if state is None:
            return
        self.state_pub.publish(state)

        # Same estimate in a standard type, so rviz/rqt_plot work off the shelf.
        odom = Odometry()
        odom.header = state.header
        odom.child_frame_id = self.object_frame
        odom.pose.pose = state.pose
        odom.twist.twist.linear = state.lin_vel
        odom.twist.twist.angular = state.ang_vel
        self.odom_pub.publish(odom)


class ObjectStateClient(object):
    """Latest `/npm/object_state`, with a staleness guard.

    A stale estimate is a frozen one, and frozen velocities read as *settled* -
    so everything here reports "not ready" past `~state_timeout_s` instead of
    letting a caller act on a dead estimator. Mirrors the estimator's
    `ready()` / `dist_dir()` API so consumers can hold either one.
    """

    def __init__(self, topic=None, timeout_s=None):
        self.topic = topic or rospy.get_param("~object_state_topic",
                                              "/npm/object_state")
        self.timeout = float(timeout_s if timeout_s is not None
                             else rospy.get_param("~state_timeout_s", 0.5))

        self.lock = threading.Lock()
        self.state = None
        # Receipt time, not the header stamp: staleness here means "this node
        # stopped hearing from the estimator", independent of sensor clocks.
        self.recv_t = None
        rospy.Subscriber(self.topic, ObjectState, self._cb, queue_size=1)
        rospy.loginfo("object_state client: %s (timeout=%.2fs)", self.topic,
                      self.timeout)

    def _cb(self, msg):
        with self.lock:
            self.state = msg
            self.recv_t = rospy.Time.now()

    def age(self):
        with self.lock:
            recv_t = self.recv_t
        if recv_t is None:
            return float("inf")
        return (rospy.Time.now() - recv_t).to_sec()

    def ready(self):
        age = self.age()
        if age > self.timeout:
            rospy.logwarn_throttle(2.0, "object_state: no fresh %s (age=%.2fs)",
                                   self.topic, age)
            return False
        return True

    def latest(self):
        """The freshest ObjectState, or None if there is none or it went stale."""
        if not self.ready():
            return None
        with self.lock:
            return self.state

    def dist_dir(self):
        state = self.latest()
        if state is None:
            return float("nan"), np.zeros(2)
        return float(state.dist_to_goal), np.array([state.dir_to_goal_x,
                                                    state.dir_to_goal_y])

    def velocities(self):
        """(lin_vel, ang_vel) in world, or None when there is no fresh state."""
        state = self.latest()
        if state is None:
            return None
        return (np.array([state.lin_vel.x, state.lin_vel.y, state.lin_vel.z]),
                np.array([state.ang_vel.x, state.ang_vel.y, state.ang_vel.z]))

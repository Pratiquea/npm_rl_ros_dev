#!/usr/bin/env python3
"""
mocap_spot_bridge_node dynamically ground Spot's internal tf tree in the
fixed mocap `world` frame.

The node recomputes the following:
    T_world_vision = T_world_spot_body_mocap        (mocap, live tf lookup)
                   . T_spot_body_mocap_body         (calib yaml, constant)
                   . inv(T_vision_body)             (spot SDK, live tf lookup)
and publishes it as a dynamic transform (world -> vision).
rosrun npm_launch mocap_spot_bridge_node.py \
      _calib_file:=$(rospack find spot_calib)/scripts/spot_mocap_calib.yaml
"""
import numpy as np
import rospy
import tf2_ros
import yaml
import geometry_msgs.msg

from spot_calib import calib_lib as cl


def tf_to_se3(tfs):
    t = tfs.transform.translation
    r = tfs.transform.rotation
    R = cl.quat_wxyz_to_rotmat([r.w, r.x, r.y, r.z])
    return cl.make_se3(R, [t.x, t.y, t.z])


def se3_to_tf(T, parent, child, stamp):
    msg = geometry_msgs.msg.TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(T[0, 3])
    msg.transform.translation.y = float(T[1, 3])
    msg.transform.translation.z = float(T[2, 3])
    q = cl.rotmat_to_quat_wxyz(T[:3, :3])   # w, x, y, z
    msg.transform.rotation.w = float(q[0])
    msg.transform.rotation.x = float(q[1])
    msg.transform.rotation.y = float(q[2])
    msg.transform.rotation.z = float(q[3])
    return msg


class Bridge(object):
    def __init__(self):
        calib_file = rospy.get_param("~calib_file")
        self.world = rospy.get_param("~world_frame", "world")
        self.mocap_body = rospy.get_param("~mocap_body_frame", "spot_body_mocap")
        self.vision = rospy.get_param("~internal_root_frame", "vision")
        # self.mode = rospy.get_param("~mode", "continuous")
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        # self.avg_samples = int(rospy.get_param("~avg_samples", 20))
        # EMA weight of the newest solve in (0, 1]; smaller = smoother. 0 = off.
        self.smoothing = float(rospy.get_param("~smoothing", 0.0))
        self.lookup_timeout = float(rospy.get_param("~lookup_timeout", 0.2))
        # Apply the calibrated mocap<->spot stream time offset (residual.td_ms).
        self.apply_time_align = bool(rospy.get_param("~apply_time_align", True))

        with open(calib_file) as f:
            calib = yaml.safe_load(f)
        ext = calib["body_extrinsic"]
        # spot body frame defaults to whatever the calib named (body / base_link).
        self.body = rospy.get_param("~internal_body_frame", ext["child_frame"])
        R = cl.quat_wxyz_to_rotmat(ext["rotation_wxyz"])
        self.T_sbm_body = cl.make_se3(R, ext["translation"])
        rospy.loginfo("mocap_spot_bridge: loaded extrinsic %s -> %s from %s",
                      ext.get("parent_frame", self.mocap_body), self.body,
                      calib_file)
        # Stream time offset: t_spot = t_mocap + td, so the mocap measurement that
        # matches a spot stamp t_ref has stamp (t_ref - td).
        try:
            self.td = float(calib["residual"]["td_ms"]) / 1000.0
        except (KeyError, TypeError):
            self.td = 0.0
            rospy.logwarn("mocap_spot_bridge: no residual.td_ms in %s; td=0",
                          calib_file)
        rospy.loginfo("mocap_spot_bridge: td=%.3f s, time_align=%s",
                      self.td, "on" if self.apply_time_align else "off")

        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf)
        self.dyn_caster = tf2_ros.TransformBroadcaster()
        self.static_caster = tf2_ros.StaticTransformBroadcaster()
        self.filt = None   # smoothed T_world_vision (continuous mode)
        self.last_stamp = None   # last published t_ref (skip duplicates)

    def solve_once(self):
        # Get (T_world_vision, t_ref).
        try:
            b = self.buf.lookup_transform(self.vision, self.body,
                                          rospy.Time(0),
                                          rospy.Duration(self.lookup_timeout))
            t_ref = b.header.stamp
            t_mocap = t_ref - rospy.Duration(self.td) \
                if self.apply_time_align else t_ref
            a = self.buf.lookup_transform(self.world, self.mocap_body,
                                          t_mocap,
                                          rospy.Duration(self.lookup_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, "mocap_spot_bridge: tf unavailable (%s)", e)
            return None, None
        T_w_sbm = tf_to_se3(a)
        T_v_b = tf_to_se3(b)
        return T_w_sbm @ self.T_sbm_body @ cl.se3_inv(T_v_b), t_ref

    def run(self):

        rospy.loginfo("mocap_spot_bridge: continuous %s -> %s @ %.1f Hz "
                      "(smoothing=%.2f)", self.world, self.vision, self.rate_hz,
                      self.smoothing)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)
        rospy.spin()

    def _tick(self, _evt):
        T, t_ref = self.solve_once()
        if T is None:
            return
        if t_ref == self.last_stamp:
            return
        self.last_stamp = t_ref
        if self.smoothing > 0.0:
            T = self._smooth(T)
        self.dyn_caster.sendTransform(
            se3_to_tf(T, self.world, self.vision, t_ref))

    def _smooth(self, T):
        """
        smoothing: translation lerp + quaternion nlerp.
        """
        if self.filt is None:
            self.filt = T
            return T
        a = self.smoothing
        t = (1.0 - a) * self.filt[:3, 3] + a * T[:3, 3]
        q0 = cl.rotmat_to_quat_wxyz(self.filt[:3, :3])
        q1 = cl.rotmat_to_quat_wxyz(T[:3, :3])
        # compute the shorter arc
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        q = (1.0 - a) * q0 + a * q1
        q = q / np.linalg.norm(q)
        self.filt = cl.make_se3(cl.quat_wxyz_to_rotmat(q), t)
        return self.filt

def main():
    rospy.init_node("mocap_spot_bridge")
    Bridge().run()


if __name__ == "__main__":
    main()

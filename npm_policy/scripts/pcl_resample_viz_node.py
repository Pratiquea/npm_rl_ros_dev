#!/usr/bin/env python3
"""
Redraw the object model from its source mesh every tick and publish it for rviz.
VISUAL ONLY: it feeds no observation and no push point.

Example:
    rosrun npm_policy pcl_resample_viz_node.py _npz_path:=<obj.npz>
"""
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

from npm_msgs.msg import ObjectState
from npm_policy import obs_lib as ol
from npm_policy import obs_viz as ov
from npm_policy import pcl_resample as pr


class PclResampleViz(object):
    def __init__(self):
        npz = rospy.get_param("~npz_path")
        self.world_frame = rospy.get_param("~world_frame", "world")
        state_topic = rospy.get_param("~object_state_topic", "/npm/object_state")
        topic = rospy.get_param("~pcl_resample/topic", "/npm/debug/resampled_cloud")
        self.enable = bool(rospy.get_param("~pcl_resample/enable", True))
        self.n_points = int(rospy.get_param("~pcl_resample/n_points", ol.N_POINTS))
        self.oversample = int(rospy.get_param("~pcl_resample/oversample", 128))
        # Own rate, not marker_rate_hz: one draw costs ~16 ms at oversample=128
        # and ~1.4 s at the npz's 10000, so this cannot ride the marker rate.
        rate = float(rospy.get_param("~pcl_resample/rate_hz", 10.0))
        mesh_path = pr.mesh_path_for(npz, rospy.get_param("~pcl_resample/mesh_path", ""))
        # A frozen pose reads as a still object, so an old state is treated as no
        # state at all rather than pinning the cloud where the object used to be.
        self.state_timeout = float(rospy.get_param("~state_timeout_s", 0.5))

        self.rng = np.random.default_rng()
        self.mesh = pr.load_mesh(mesh_path)
        self.state = None
        self.cloud_pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
        rospy.Subscriber(state_topic, ObjectState, self._state_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / rate), self._tick)
        rospy.loginfo("pcl_resample_viz: %s -> %s at %.1f Hz (enable=%s n_points=%d "
                      "oversample=%d); VISUAL ONLY", mesh_path, topic, rate,
                      self.enable, self.n_points, self.oversample)

    def _state_cb(self, msg):
        self.state = msg

    def _tick(self, _event):
        msg = self.state
        if not self.enable or msg is None:
            return
        age = (rospy.Time.now() - msg.header.stamp).to_sec()
        if msg.header.stamp != rospy.Time(0) and age > self.state_timeout:
            rospy.logwarn_throttle(5.0, "pcl_resample_viz: object state %.2f s old, "
                                        "not drawing", age)
            return

        try:
            pts, _nrm = pr.resample_cloud(self.mesh, self.n_points, self.oversample,
                                          self.rng)
        except Exception as e:
            rospy.logerr_throttle(5.0, "pcl_resample_viz: resample failed: %s", e)
            return

        # Published already rotated and translated into `world`. The object_obs
        # convention would need a TF broadcast, and that frame already has a
        # publisher (policy_node / test_viz_obs_cloud); a second one makes tf2
        # flip its parent per sample.
        rot = ol.rotation_from_ros_quat(msg.pose.orientation)
        p = msg.pose.position
        world = np.asarray(rot.apply(pts)) + np.array([p.x, p.y, p.z])
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) \
            else rospy.Time.now()
        self.cloud_pub.publish(
            ov.cloud_msg(world.reshape(-1), stamp, frame=self.world_frame))


def main():
    rospy.init_node("pcl_resample_viz")
    PclResampleViz()
    rospy.spin()


if __name__ == "__main__":
    main()

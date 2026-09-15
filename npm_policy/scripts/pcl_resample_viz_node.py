#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

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
        rate = float(rospy.get_param("~pcl_resample/rate_hz", 10.0))
        self.mesh_path = pr.mesh_path_for(npz,
                                          rospy.get_param("~pcl_resample/mesh_path", ""))
        self.state_timeout = float(rospy.get_param("~state_timeout_s", 0.5))

        self.mesh_enable = bool(rospy.get_param("~mesh_marker/enable", True))
        self.mesh_mode = str(rospy.get_param("~mesh_marker/mode", "resource"))
        mesh_rate = float(rospy.get_param("~mesh_marker/rate_hz",
                                          rospy.get_param("~marker_rate_hz", 20.0)))
        mesh_topic = rospy.get_param("~mesh_marker/topic", "/npm/debug/object_mesh")
        self.mesh_color = self._color_param("~mesh_marker/color", ov.MESH_COLOR)
        self.mesh_ns = rospy.get_param("~mesh_marker/ns", "object_mesh")

        self.rng = np.random.default_rng()
        self.mesh = pr.load_mesh(self.mesh_path)
        self.state = None
        self.mesh_drawn = False
        self.tris = None
        if self.mesh_enable and self.mesh_mode == "triangles":
            self.tris = pr.mesh_triangles(self.mesh)

        self.cloud_pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
        self.mesh_pub = rospy.Publisher(mesh_topic, Marker, queue_size=1)
        rospy.Subscriber(state_topic, ObjectState, self._state_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / rate), self._tick)
        if self.mesh_enable:
            rospy.Timer(rospy.Duration(1.0 / mesh_rate), self._mesh_tick)
        rospy.loginfo("pcl_resample_viz: %s -> %s at %.1f Hz (enable=%s n_points=%d "
                      "oversample=%d); mesh -> %s at %.1f Hz (enable=%s mode=%s); ",
                       self.mesh_path, topic, rate, self.enable,
                      self.n_points, self.oversample, mesh_topic, mesh_rate,
                      self.mesh_enable, self.mesh_mode)

    @staticmethod
    def _color_param(name, default):
        rgba = rospy.get_param(name, None)
        if not rgba:
            return default
        return ColorRGBA(*[float(v) for v in rgba])

    def _state_cb(self, msg):
        self.state = msg

    def _fresh_state(self, what):
        msg = self.state
        if msg is None:
            return None
        age = (rospy.Time.now() - msg.header.stamp).to_sec()
        if msg.header.stamp != rospy.Time(0) and age > self.state_timeout:
            rospy.logwarn_throttle(5.0, "pcl_resample_viz: object state %.2f s old, "
                                        "not drawing %s", age, what)
            return None
        return msg

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp if msg.header.stamp != rospy.Time(0) \
            else rospy.Time.now()

    def _tick(self, _event):
        if not self.enable:
            return
        msg = self._fresh_state("the cloud")
        if msg is None:
            return

        try:
            pts, _nrm = pr.resample_cloud(self.mesh, self.n_points, self.oversample,
                                          self.rng)
        except Exception as e:
            rospy.logerr_throttle(5.0, "pcl_resample_viz: resample failed: %s", e)
            return

        rot = ol.rotation_from_ros_quat(msg.pose.orientation)
        p = msg.pose.position
        world = np.asarray(rot.apply(pts)) + np.array([p.x, p.y, p.z])
        self.cloud_pub.publish(
            ov.cloud_msg(world.reshape(-1), self._stamp(msg), frame=self.world_frame))

    def _mesh_tick(self, _event):
        msg = self._fresh_state("the mesh")
        if msg is None:
            if self.mesh_drawn:
                self.mesh_pub.publish(
                    ov.mesh_delete_marker(rospy.Time.now(), frame=self.world_frame,
                                          ns=self.mesh_ns))
                self.mesh_drawn = False
            return

        stamp = self._stamp(msg)
        if self.tris is not None:
            marker = ov.mesh_triangles_marker(self.tris[0], self.tris[1], msg.pose,
                                              stamp, frame=self.world_frame,
                                              color=self.mesh_color, ns=self.mesh_ns)
        else:
            marker = ov.mesh_resource_marker(self.mesh_path, msg.pose, stamp,
                                             frame=self.world_frame,
                                             color=self.mesh_color, ns=self.mesh_ns)
        self.mesh_pub.publish(marker)
        self.mesh_drawn = True


def main():
    rospy.init_node("pcl_resample_viz")
    PclResampleViz()
    rospy.spin()


if __name__ == "__main__":
    main()

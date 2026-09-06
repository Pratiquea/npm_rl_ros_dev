#!/usr/bin/env python3
"""
Publish two clouds at full mocap rate so the obs pipeline can be checked live in
rviz while hand-moving the object. No filtering, no policy, no robot.

  /npm/debug/obs_cloud     LINK-frame cloud (pts_norm*scale + centroid) rotated
                           and placed on the object: true size, checks rotation.
  /npm/debug/raw_obs_cloud the literal obs[0:384] / obs[384:768] blocks that
                           policy_node feeds the network, parked at the goal.

Example:
    rosrun npm_policy test_viz_obs_cloud.py _npz_path:=<obj.npz>
"""
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from npm_msgs.msg import ObjectState

from npm_policy import obs_lib as ol
from npm_policy import obs_viz as ov


class ObsCloudDebug(object):
    def __init__(self):
        npz = rospy.get_param("~npz_path")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.state_topic = rospy.get_param("~object_state_topic",
                                           "/npm/object_state")
        self.normal_len = float(rospy.get_param("~normal_marker_len", 0.05))
        # Extras-block fields only: they cannot move a point, they are here so the
        # assembled vector is the same one policy_node would build.
        self.mass = float(rospy.get_param("~object_mass", 20.0))
        self.friction = float(rospy.get_param("~friction", 0.95))
        self.raw_origin = list(rospy.get_param("~raw_obs_origin", [0.0, 0.0, 0.0]))

        # Same margin the policy resolves loc_idx with, so this node and
        # policy_node draw the identical pair of clouds.
        self.shrink = float(rospy.get_param("~pcl_shrink", 1.0))

        self.pts, self.normals, self.scale, self.centroid = ol.load_model_npz(npz)
        # obs_cloud is the NOMINAL model, obs_cloud_shrunk is what the executor
        # aims at. The gap between the two is the pcl_shrink margin: it is only a
        # few mm, so it needs both clouds and small rviz points to be visible.
        self.link_pts = ol.link_frame_cloud(self.pts, self.scale, self.centroid)
        self.link_pts_shrunk = ol.link_frame_cloud(self.pts, self.scale,
                                                   self.centroid, self.shrink)

        self.tf_caster = tf2_ros.TransformBroadcaster()
        self.cloud_pub = rospy.Publisher("/npm/debug/obs_cloud", PointCloud2,
                                         queue_size=1)
        self.shrunk_pub = rospy.Publisher("/npm/debug/obs_cloud_shrunk", PointCloud2,
                                          queue_size=1)
        self.normals_pub = rospy.Publisher("/npm/debug/obs_normals", Marker,
                                           queue_size=1)
        self.raw_cloud_pub = rospy.Publisher("/npm/debug/raw_obs_cloud", PointCloud2,
                                             queue_size=1)
        self.raw_normals_pub = rospy.Publisher("/npm/debug/raw_obs_normals", Marker,
                                               queue_size=1)
        rospy.Subscriber(self.state_topic, ObjectState, self._state_cb,
                         queue_size=1)

        # This node runs exactly when policy_node does not: npm_stack gates it on
        # obs_viz = (not policy). So it is the only /npm/model_info publisher in
        # the rigs that have no policy - the mocap-only object_state.launch, and
        # npm_stack policy:=false, where executor_node subscribes and would
        # otherwise wait forever for a model it can never verify.
        self.model_pub = rospy.Publisher("/npm/model_info", String, queue_size=1,
                                         latch=True)
        self.model_pub.publish(
            String(ol.format_model_info(npz, self.scale, self.shrink)))

        inset_mm = 1000.0 * ol.shrink_inset(self.pts, self.scale, self.shrink)
        rospy.loginfo("test_viz_obs_cloud: %s -> %s (scale=%.4f pcl_shrink=%.4f, "
                      "inset %.1f/%.1f/%.1f mm); obs_cloud is NOMINAL, "
                      "obs_cloud_shrunk is the margin; publishing /npm/model_info "
                      "(no policy_node in this rig)",
                      self.state_topic, self.world_frame, self.scale, self.shrink,
                      inset_mm[0], inset_mm[1], inset_mm[2])

    def _state_cb(self, msg):
        # object_state_node already resolved the pose into world, so this is the
        # same rotation the policy applies - no TF hop of our own.
        rot = ol.rotation_from_ros_quat(msg.pose.orientation)
        p = msg.pose.position
        stamp = msg.header.stamp or rospy.Time.now()

        # Normals are unit directions: rotated, never scaled or translated. They
        # are drawn from the nominal points, so they start on the real surface.
        tf, cloud, normals = ov.cloud_and_normals(
            ol.rotate_cloud(rot, self.link_pts),
            ol.rotate_cloud(rot, self.normals),
            [p.x, p.y, p.z], stamp, parent=self.world_frame,
            length=self.normal_len)
        self.tf_caster.sendTransform(tf)
        self.cloud_pub.publish(cloud)
        self.normals_pub.publish(normals)
        # Same frame and same stamp, so the two clouds always overlay.
        self.shrunk_pub.publish(
            ov.cloud_msg(ol.rotate_cloud(rot, self.link_pts_shrunk), stamp))
        self._publish_raw_obs(rot, [p.x, p.y, p.z], msg, stamp)

    def _publish_raw_obs(self, rot, pos, msg, stamp):
        # Build the full 798-vector through the same function policy_node's
        # infer_push calls, then draw the blocks straight out of it - no local
        # rotate_cloud, so a change to the obs layout cannot pass unseen here.
        obs = ol.assemble_observation(
            rot, pos,
            [msg.lin_vel.x, msg.lin_vel.y, msg.lin_vel.z],
            [msg.ang_vel.x, msg.ang_vel.y, msg.ang_vel.z],
            np.zeros(4), self.mass, self.friction, self.scale,
            self.pts, self.normals)

        tf, cloud, normals = ov.raw_obs_cloud_and_normals(
            obs, self.raw_origin, stamp, parent=self.world_frame,
            length=self.normal_len)
        self.tf_caster.sendTransform(tf)
        self.raw_cloud_pub.publish(cloud)
        self.raw_normals_pub.publish(normals)


def main():
    rospy.init_node("test_viz_obs_cloud")
    ObsCloudDebug()
    rospy.spin()


if __name__ == "__main__":
    main()

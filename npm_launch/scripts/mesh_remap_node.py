#!/usr/bin/env python3
import os

import rospy
from visualization_msgs.msg import Marker, MarkerArray


class MeshRemap(object):
    def __init__(self):
        self.package = rospy.get_param("~package", "npm_launch")
        self.models_dir = rospy.get_param("~models_dir", "models")
        queue = int(rospy.get_param("~queue_size", 10))

        marker_in = rospy.get_param("~marker_in", "/npm/debug/object_mesh_bag")
        marker_out = rospy.get_param("~marker_out", "/npm/debug/object_mesh")
        array_in = rospy.get_param("~array_in", "/npm/debug/settled_snapshots_bag")
        array_out = rospy.get_param("~array_out", "/npm/debug/settled_snapshots")

        self.seen = set()
        self.marker_pub = rospy.Publisher(marker_out, Marker, queue_size=queue)
        self.array_pub = rospy.Publisher(array_out, MarkerArray, queue_size=queue)
        rospy.Subscriber(marker_in, Marker, self.on_marker, queue_size=queue)
        rospy.Subscriber(array_in, MarkerArray, self.on_array, queue_size=queue)

        rospy.loginfo("mesh_remap: %s -> %s, %s -> %s",
                      marker_in, marker_out, array_in, array_out)

    def rewrite(self, resource):
        if not resource or resource.startswith("package://"):
            return resource
        name = os.path.basename(resource)
        stem = os.path.splitext(name)[0]
        out = "package://%s/%s/%s/%s" % (self.package, self.models_dir, stem, name)
        if resource not in self.seen:
            self.seen.add(resource)
            rospy.loginfo("mesh_remap: rewrote %s to %s", resource, out)
        return out

    def on_marker(self, msg):
        msg.mesh_resource = self.rewrite(msg.mesh_resource)
        self.marker_pub.publish(msg)

    def on_array(self, msg):
        for marker in msg.markers:
            marker.mesh_resource = self.rewrite(marker.mesh_resource)
        self.array_pub.publish(msg)


def main():
    rospy.init_node("mesh_remap")
    MeshRemap()
    rospy.spin()


if __name__ == "__main__":
    main()

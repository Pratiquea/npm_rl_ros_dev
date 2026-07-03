#!/usr/bin/env python3
"""test_push_publisher: publish one PushCommand to drive executor_node.
    rosrun npm_control test_push_publisher.py --force-mag 30 --loc-idx 7
"""
import argparse
import sys

import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs 
from geometry_msgs.msg import PointStamped, Vector3Stamped, PoseStamped
from spot_msgs.msg import FootStateArray
from npm_msgs.msg import PushCommand

# /spot/status/feet ordering
# 0 front-left, 1 front-right, 2 rear-left, 3 rear-right.
FRONT_LEFT_FOOT = 0
FRONT_RIGHT_FOOT = 1
FORWARD_OFFSET_M = 0.3  #distance from the front feet (body +x)


def _object_x_axis(q):
    #Unit x-axis of a frame from its xyzw quaternion.
    x, y, z, w = q
    return np.array([1.0 - 2.0 * (y * y + z * z),
                     2.0 * (x * y + w * z),
                     2.0 * (x * z - w * y)])


def _contact_from_object(pose_topic, target_frame, tf_buffer, timeout_s):
    pose = rospy.wait_for_message(pose_topic, PoseStamped, timeout=timeout_s)
    src_frame = pose.header.frame_id
    p, o = pose.pose.position, pose.pose.orientation

    # Object origin as a point in the pose's source frame.
    pt = PointStamped()
    pt.header.frame_id = src_frame
    pt.header.stamp = rospy.Time(0)
    pt.point.x, pt.point.y, pt.point.z = p.x, p.y, p.z

    # Object x-axis as a vector in the pose's source frame.
    xaxis = _object_x_axis([o.x, o.y, o.z, o.w])
    vec = Vector3Stamped()
    vec.header.frame_id = src_frame
    vec.header.stamp = rospy.Time(0)
    vec.vector.x, vec.vector.y, vec.vector.z = xaxis

    tf = tf_buffer.lookup_transform(target_frame, src_frame, rospy.Time(0),
                                    rospy.Duration(timeout_s))
    out_pt = tf2_geometry_msgs.do_transform_point(pt, tf)
    out_vec = tf2_geometry_msgs.do_transform_vector3(vec, tf)

    contact = np.array([out_pt.point.x, out_pt.point.y, out_pt.point.z])
    direction = np.array([out_vec.vector.x, out_vec.vector.y, out_vec.vector.z])
    n = np.linalg.norm(direction)
    if n > 1e-9:
        direction = direction / n 
    return contact, direction


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force-mag", type=float, default=5.0, dest="force_mag",
                        help="push force magnitude (N), applied along object_1 +x")
    parser.add_argument("--loc-idx", type=int, default=0, dest="loc_idx",
                        help="point-cloud index (logging)")
    parser.add_argument("--object-topic", default="/mocap_node/object_1/pose",
                        dest="object_topic", help="mocap object_1 PoseStamped topic")
    parser.add_argument("--feet-topic", default="/spot/status/feet", dest="feet_topic",
                        help="spot_ros FootStateArray topic (unused; legacy contact mode)")
    parser.add_argument("--body-frame", default="body", dest="body_frame",
                        help="spot_ros body TF frame (unused; legacy contact mode)")
    parser.add_argument("--odom-frame", default="odom", dest="odom_frame",
                        help="frame the PushCommand is published in")
    parser.add_argument("--tf-timeout", type=float, default=5.0, dest="tf_timeout",
                        help="seconds to wait for feet msg / TF")
    config = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("test_push_publisher")
    pub = rospy.Publisher("/npm/push_command_active", PushCommand, queue_size=1, latch=True)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)  # noqa: F841 (keeps buffer fed)

    # Contact = object_1 origin, push direction = object_1 x-axis, both in odom.
    contact, direction = _contact_from_object(config.object_topic, config.odom_frame,
                                              tf_buffer, config.tf_timeout)
    force = direction * config.force_mag

    msg = PushCommand()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = config.odom_frame
    msg.loc_idx = config.loc_idx
    msg.contact_point.x, msg.contact_point.y, msg.contact_point.z = contact
    msg.push_force.x, msg.push_force.y, msg.push_force.z = force

    # Wait for the subscriber to connect.
    rospy.loginfo("Waiting for a subscriber on /npm/push_command_active...")
    rate = rospy.Rate(10)
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rate.sleep()

    pub.publish(msg)
    rospy.loginfo("Published PushCommand: contact=%s force=%s frame=%s loc_idx=%d",
                  contact.tolist(), force.tolist(), config.odom_frame, config.loc_idx)

    rospy.sleep(1.0)


if __name__ == "__main__":
    main()

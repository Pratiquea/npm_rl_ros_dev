#!/usr/bin/env python3
"""
test_push_action_static: tests executor_node with one ExecutePush goal without 
the presence of an object. The goal/object point is defined spot body frame for 
convenience and then transformed to the object frame for ExecutePushGoal.

rosrun npm_control test_push_action_static.py --force-mag 30 --object-pos 1.0 0.0 0.3
"""
import argparse
import sys

import rospy
import actionlib
import tf2_ros
import tf2_geometry_msgs 
from geometry_msgs.msg import TransformStamped, Point, PoseStamped, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from npm_msgs.msg import ExecutePushAction, ExecutePushGoal
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME

_PHASE = {0: "ALIGN", 1: "APPROACH", 2: "PUSH"}


def _feedback_cb(fb):
    rospy.loginfo_throttle(0.25, "  [%s] contact=%s F=%.1fN drift=%.3fm",
                           _PHASE.get(fb.phase, "?"), fb.contact_made, fb.cur_force,
                           fb.point_drift)


def _make_markers(frame, point, force, force_mag):
    stamp = rospy.Time.now()

    sphere = Marker()
    sphere.header.frame_id = frame
    sphere.header.stamp = stamp
    sphere.ns = "push_test"
    sphere.id = 0
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose.position = point
    sphere.pose.orientation.w = 1.0
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.03
    sphere.color = ColorRGBA(1.0, 0.2, 0.2, 1.0)

    arrow = Marker()
    arrow.header.frame_id = frame
    arrow.header.stamp = stamp
    arrow.ns = "push_test"
    arrow.id = 1
    arrow.type = Marker.ARROW
    arrow.action = Marker.ADD
    arrow.pose.orientation.w = 1.0
    # 0.05 m per N so arrow length scales with commanded force.
    tip = Point(point.x + force.x * 0.05,
                point.y + force.y * 0.05,
                point.z + force.z * 0.05)
    arrow.points = [point, tip]
    arrow.scale.x = 0.01   # shaft diameter
    arrow.scale.y = 0.02   # head diameter
    arrow.scale.z = 0.03   # head length
    arrow.color = ColorRGBA(0.2, 0.6, 1.0, 1.0)

    arr = MarkerArray()
    arr.markers = [sphere, arrow]
    return arr


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force-mag", type=float, default=10.0, dest="force_mag",
                        help="push force magnitude (N), along object +x")
    parser.add_argument("--loc-idx", type=int, default=0, dest="loc_idx",
                        help="point-cloud index (logging)")
    parser.add_argument("--duration", type=float, default=3.0, help="push duration (s)")
    parser.add_argument("--object-frame", default="object", dest="object_frame",
                        help="object TF frame the executor tracks (match the executor)")
    parser.add_argument("--root-frame", default="odom", dest="root_frame",
                        help="parent frame for the static object TF")
    parser.add_argument("--object-pos", type=float, nargs=3, default=[1.0, 0.0, 0.3],
                        dest="object_pos", metavar=("X", "Y", "Z"),
                        help="object position [x y z], in --object-in-frame")
    parser.add_argument("--object-in-frame", default=BODY_FRAME_NAME, dest="object_in_frame",
                        help="frame --object-pos is given in (transformed to odom)")
    config = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("test_push_action_static")

    root_frame = None
    if config.root_frame == "odom":
        root_frame = ODOM_FRAME_NAME
    elif config.root_frame == "vision":
        root_frame = VISION_FRAME_NAME
    else:
        rospy.logerr("Unknown root frame {}, exiting.".format(config.root_frame))
        return

    # Object pose is given in --object-in-frame (like body); transform to odom.
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)
    src = PoseStamped()
    src.header.frame_id = config.object_in_frame
    src.header.stamp = rospy.Time(0)  # latest available
    src.pose.position.x, src.pose.position.y, src.pose.position.z = config.object_pos
    src.pose.orientation.w = 1.0
    try:
        dst = buf.transform(src, root_frame, timeout=rospy.Duration(5.0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
        rospy.logerr("Transform %s->%s failed: %s", config.object_in_frame, root_frame, e)
        return
    p, q = dst.pose.position, dst.pose.orientation

    # Static object pose in odom (position + orientation both transformed from body).
    bc = tf2_ros.StaticTransformBroadcaster()
    tf = TransformStamped()
    tf.header.stamp = rospy.Time.now()
    tf.header.frame_id = root_frame
    tf.child_frame_id = config.object_frame
    tf.transform.translation.x = p.x
    tf.transform.translation.y = p.y
    tf.transform.translation.z = p.z
    tf.transform.rotation = q
    bc.sendTransform(tf)
    rospy.loginfo("Object %s(%.2f,%.2f,%.2f) -> %s(%.2f,%.2f,%.2f); static TF %s<-%s",
                  config.object_in_frame, config.object_pos[0], config.object_pos[1],
                  config.object_pos[2], root_frame, p.x, p.y, p.z,
                  root_frame, config.object_frame)

    goal = ExecutePushGoal()
    goal.loc_idx = config.loc_idx
    goal.push_point = Point(0.0, 0.0, 0.0)                 # object origin
    goal.force_body = Vector3(config.force_mag, 0.0, 0.0)  # object +x
    goal.duration = config.duration
    goal.root_frame = ""

    # Visualize the push point and dir
    marker_pub = rospy.Publisher("push_test_marker", MarkerArray, queue_size=1, latch=True)
    markers = _make_markers(config.object_frame, goal.push_point, goal.force_body,
                            config.force_mag)
    marker_pub.publish(markers)
    rospy.loginfo("Published push marker on 'push_test_marker' (frame=%s). "
                  "Inspect in rviz.", config.object_frame)

    # need user confirmation before commanding any motion.
    try:
        ans = input("Send goal? [y/N] ") 
    except NameError:
        ans = input("Send goal? [y/N] ")
    if ans.strip().lower() not in ("y", "yes"):
        rospy.loginfo("Aborted by user; no goal sent.")
        return

    client = actionlib.SimpleActionClient("push", ExecutePushAction)
    rospy.loginfo("Waiting for ExecutePush server...")
    client.wait_for_server()

    rospy.loginfo("Sending goal: |F|=%.1fN dur=%.1fs", config.force_mag, config.duration)
    client.send_goal(goal, feedback_cb=_feedback_cb)
    client.wait_for_result()
    res = client.get_result()
    rospy.loginfo("RESULT: contact=%s peak=%.1fN drift=%.3fm reason=%s",
                  res.contact_made, res.peak_force, res.finger_travel, res.end_reason)


if __name__ == "__main__":
    main()

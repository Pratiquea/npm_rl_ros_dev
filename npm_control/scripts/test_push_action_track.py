#!/usr/bin/env python3
"""
test_push_action_track: tests executor_node with one ExecutePush goal and real object.
rosrun npm_control test_push_action_track.py --force-mag 30
"""
import argparse
import sys

import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, Point, Vector3
from npm_msgs.msg import ExecutePushAction, ExecutePushGoal

_PHASE = {0: "ALIGN", 1: "APPROACH", 2: "PUSH"}


def _feedback_cb(fb):
    rospy.loginfo_throttle(0.25, "  [%s] contact=%s F=%.1fN drift=%.3fm",
                           _PHASE.get(fb.phase, "?"), fb.contact_made, fb.cur_force,
                           fb.point_drift)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force-mag", type=float, default=10.0, dest="force_mag",
                        help="push force magnitude (N), along object +x")
    parser.add_argument("--loc-idx", type=int, default=0, dest="loc_idx",
                        help="point-cloud index (logging)")
    parser.add_argument("--duration", type=float, default=3.0, help="push duration (s)")
    parser.add_argument("--push-point", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        dest="push_point", metavar=("X", "Y", "Z"),
                        help="locked surface point in the OBJECT frame (default origin)")
    parser.add_argument("--object-frame", default="object_1", dest="object_frame",
                        help="object TF frame the executor tracks the locked push point in")
    parser.add_argument("--object-topic", default="/mocap_node/object_1/pose",
                        dest="object_topic", help="mocap object PoseStamped topic (liveness check)")
    parser.add_argument("--tf-timeout", type=float, default=5.0, dest="tf_timeout",
                        help="seconds to wait for the mocap pose message")
    config = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("test_push_action_track")

    # Check if mocap is live.
    try:
        rospy.wait_for_message(config.object_topic, PoseStamped, timeout=config.tf_timeout)
        rospy.loginfo("Object pose is live on %s.", config.object_topic)
    except rospy.ROSException:
        rospy.logwarn("No object pose on %s within %.1fs — sending anyway; executor needs "
                      "the object TF to track.", config.object_topic, config.tf_timeout)

    client = actionlib.SimpleActionClient("push", ExecutePushAction)
    rospy.loginfo("Waiting for ExecutePush server...")
    client.wait_for_server()

    goal = ExecutePushGoal()
    goal.loc_idx = config.loc_idx
    goal.push_point = Point(*[float(v) for v in config.push_point])  # OBJECT frame
    goal.force_body = Vector3(config.force_mag, 0.0, 0.0)            # object +x
    goal.duration = config.duration
    goal.root_frame = ""
    goal.object_frame = config.object_frame

    rospy.loginfo("Sending goal: |F|=%.1fN dur=%.1fs push_point=%s (object frame)",
                  config.force_mag, config.duration, config.push_point)
    client.send_goal(goal, feedback_cb=_feedback_cb)
    client.wait_for_result()
    res = client.get_result()
    rospy.loginfo("RESULT: contact=%s peak=%.1fN drift=%.3fm reason=%s",
                  res.contact_made, res.peak_force, res.finger_travel, res.end_reason)


if __name__ == "__main__":
    main()

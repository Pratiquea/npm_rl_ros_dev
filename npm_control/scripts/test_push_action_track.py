#!/usr/bin/env python3
"""
Send one ExecutePush goal against the live mocap-tracked object.

Two ways to name the contact point, matching the executor's two paths:
  --loc-idx N     point-cloud index, the sim/policy path (executor resolves it)
  --push-point    explicit object-frame point, the bench path (loc_idx sent as -1)

Example:
    rosrun npm_control test_push_action_track.py --loc-idx 5 --force-along-normal
    rosrun npm_control test_push_action_track.py --push-point 0 0 0 --force-mag 30
"""
import argparse
import os
import sys

import numpy as np
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, Point, Vector3
from npm_msgs.msg import ExecutePushAction, ExecutePushGoal

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "npm_policy", "src"))
from npm_policy import obs_lib as ol

_PHASE = {0: "ALIGN", 1: "APPROACH", 2: "PUSH", 3: "RETREAT", 4: "TOPPLE"}
_last_phase = {"v": None}    # holder so the cb can detect phase transitions


def _feedback_cb(fb):
    if fb.phase != _last_phase["v"]:
        rospy.loginfo("PHASE -> %s", _PHASE.get(fb.phase, "?"))
        _last_phase["v"] = fb.phase
    rospy.loginfo_throttle(0.25, "  [%s] contact=%s F=%.1fN drift=%.3fm",
                           _PHASE.get(fb.phase, "?"), fb.contact_made, fb.cur_force,
                           fb.point_drift)


def _resolve_npz_path(cli_path):
    # The executor is the node that really needs the model, so fall back to its
    # param: running both off one path is the point of the agreement check.
    if cli_path:
        return cli_path
    return rospy.get_param("~npz_path",
                           rospy.get_param("/executor_node/npz_path", ""))


# Nodes that hold a pcl_shrink, in the order they should be trusted: the
# executor is the one that resolves loc_idx for real, then the policy that hands
# it the index, then the mocap-only viz rig, which is the only one of the three
# running under object_state.launch.
_SHRINK_SOURCES = ("/executor_node/pcl_shrink",
                   "/policy_node/pcl_shrink",
                   "/test_viz_obs_cloud/pcl_shrink")


def _resolve_shrink(cli_shrink):
    # Mirror the running stack's contact-cloud margin; a local default would draw
    # a cloud the robot never aims at. Which node supplies it depends on the rig,
    # so the source is logged rather than left to be guessed from the geometry.
    if cli_shrink is not None:
        rospy.loginfo("pcl_shrink=%.4f (--shrink)", float(cli_shrink))
        return float(cli_shrink)
    if rospy.has_param("~pcl_shrink"):
        shrink = float(rospy.get_param("~pcl_shrink"))
        rospy.loginfo("pcl_shrink=%.4f (~pcl_shrink)", shrink)
        return shrink
    for name in _SHRINK_SOURCES:
        if rospy.has_param(name):
            shrink = float(rospy.get_param(name))
            rospy.loginfo("pcl_shrink=%.4f (from %s)", shrink, name)
            return shrink
    rospy.logwarn("No pcl_shrink on the param server (%s); drawing the NOMINAL "
                  "cloud. If the stack runs a margin, this cloud is not the one "
                  "it aims at.", ", ".join(_SHRINK_SOURCES))
    return 1.0


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force-mag", type=float, default=10.0, dest="force_mag",
                        help="push force magnitude (N)")
    parser.add_argument("--loc-idx", type=int, default=-1, dest="loc_idx",
                        help="point-cloud index [0,128); the executor resolves the "
                             "contact point from its own model. <0 (default) means "
                             "use --push-point instead")
    parser.add_argument("--duration", type=float, default=3.0, help="push duration (s)")
    parser.add_argument("--push-point", type=float, nargs=3, default=None,
                        dest="push_point", metavar=("X", "Y", "Z"),
                        help="explicit locked surface point in the OBJECT frame "
                             "(default origin); mutually exclusive with --loc-idx")
    parser.add_argument("--force-dir", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        dest="force_dir", metavar=("X", "Y", "Z"),
                        help="push direction in the OBJECT frame (default object +z)")
    parser.add_argument("--force-along-normal", action="store_true",
                        dest="force_along_normal",
                        help="push along -normal at --loc-idx (needs the model npz). "
                             "The sim's cone parametrization keeps every push within "
                             "20 deg of -normal; a "
                             "fixed --force-dir at an arbitrary index often grazes "
                             "the surface or pulls away from it")
    parser.add_argument("--npz-path", default="", dest="npz_path",
                        help="object model .npz (default: the executor's ~npz_path)")
    parser.add_argument("--shrink", type=float, default=None,
                        help="contact-cloud margin (default: the executor's "
                             "~pcl_shrink); must match it or the expected point "
                             "will not be the commanded one")
    parser.add_argument("--object-frame", default="object_link", dest="object_frame",
                        help="object TF frame the executor tracks the push point in")
    parser.add_argument("--object-topic", default="", dest="object_topic",
                        help="mocap object PoseStamped topic (default: object_state_node's ~object_pose_topic)")
    parser.add_argument("--tf-timeout", type=float, default=5.0, dest="tf_timeout",
                        help="seconds to wait for the mocap pose message")
    config = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    if config.loc_idx >= 0 and config.push_point is not None:
        parser.error("--loc-idx and --push-point name the same thing two different "
                     "ways; pass exactly one.")
    if config.force_along_normal and config.loc_idx < 0:
        parser.error("--force-along-normal needs --loc-idx to pick a normal.")
    return config


def _build_force(config):
    # Object-frame force vector, plus the model point the executor should land on
    # (loc_idx mode only, for eyeballing against the executor's GOAL log).
    expected = None
    direction = np.asarray(config.force_dir, dtype=np.float64)
    if config.loc_idx >= 0:
        npz = _resolve_npz_path(config.npz_path)
        if npz:
            pts, normals, scale, centroid = ol.load_model_npz(npz)
            expected = ol.link_frame_cloud(pts, scale, centroid,
                                           _resolve_shrink(config.shrink))[config.loc_idx]
            if config.force_along_normal:
                direction = -np.asarray(normals[config.loc_idx], dtype=np.float64)
        elif config.force_along_normal:
            rospy.logfatal("--force-along-normal needs a model: pass --npz-path or "
                           "set /executor_node/npz_path.")
            sys.exit(1)
    n = float(np.linalg.norm(direction))
    if n < 1e-9:
        rospy.logfatal("--force-dir is zero-length.")
        sys.exit(1)
    return direction / n * config.force_mag, expected


def _resolve_object_topic(cli_topic):
    # Same mocap body as the running stack, or this script watches an object
    # nobody is pushing. Mirrors debug_loc_idx_viz's npz_path fallback.
    if cli_topic:
        return cli_topic
    return rospy.get_param("/object_state_node/object_pose_topic",
                           "/object/pose")


def main():
    config = _parse_args()
    rospy.init_node("test_push_action_track")
    config.object_topic = _resolve_object_topic(config.object_topic)

    # Check if mocap is live.
    try:
        rospy.wait_for_message(config.object_topic, PoseStamped, timeout=config.tf_timeout)
        rospy.loginfo("Object pose is live on %s.", config.object_topic)
    except rospy.ROSException:
        rospy.logwarn("No object pose on %s within %.1fs - sending anyway; executor needs "
                      "the object TF to track.", config.object_topic, config.tf_timeout)

    force, expected = _build_force(config)

    client = actionlib.SimpleActionClient("push", ExecutePushAction)
    rospy.loginfo("Waiting for ExecutePush server...")
    client.wait_for_server()

    goal = ExecutePushGoal()
    goal.loc_idx = config.loc_idx
    # In loc_idx mode the executor owns the resolution, so leave push_point zero
    # rather than sending a second opinion it would have to reconcile.
    pt = config.push_point if config.loc_idx < 0 else None
    goal.push_point = Point(*[float(v) for v in (pt or [0.0, 0.0, 0.0])])
    goal.force_body = Vector3(*[float(v) for v in force])
    goal.duration = config.duration
    goal.root_frame = ""
    goal.object_frame = config.object_frame

    if config.loc_idx >= 0:
        rospy.loginfo("Sending goal: loc_idx=%d |F|=%.1fN dir=(%.2f,%.2f,%.2f) dur=%.1fs",
                      config.loc_idx, config.force_mag, force[0] / config.force_mag,
                      force[1] / config.force_mag, force[2] / config.force_mag,
                      config.duration)
        if expected is not None:
            rospy.loginfo("  model point at loc %d = (%.3f, %.3f, %.3f) in %s; the "
                          "executor's GOAL log should match.", config.loc_idx,
                          expected[0], expected[1], expected[2], config.object_frame)
    else:
        rospy.loginfo("Sending goal: push_point=(%.3f, %.3f, %.3f) (object frame) "
                      "|F|=%.1fN dur=%.1fs",
                      goal.push_point.x, goal.push_point.y, goal.push_point.z,
                      config.force_mag, config.duration)

    client.send_goal(goal, feedback_cb=_feedback_cb)
    client.wait_for_result()
    res = client.get_result()
    rospy.loginfo("RESULT: contact=%s peak=%.1fN drift=%.3fm reason=%s",
                  res.contact_made, res.peak_force, res.finger_travel, res.end_reason)


if __name__ == "__main__":
    main()

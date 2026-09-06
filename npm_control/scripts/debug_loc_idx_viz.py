#!/usr/bin/env python3
"""
Show which model point a loc_idx names, in rviz, without moving the robot.

Pick an index three ways while the node stays up: type it at the prompt, publish
it on ~loc_idx, or click the object with rviz's "Publish Point" tool to get the
nearest index back. Use it to choose the --loc-idx for test_push_action_track.

Example:
    rosrun npm_control debug_loc_idx_viz.py
    rostopic pub -1 /debug_loc_idx_viz/loc_idx std_msgs/Int32 "data: 42"
"""
import argparse
import sys
import threading

import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs    # registers PointStamped with the tf2 buffer
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import ColorRGBA, Int32
from visualization_msgs.msg import Marker, MarkerArray

from npm_control import executor_lib as el
from npm_policy import obs_lib as ol

_NS = "loc_idx_probe"
_ID_CLOUD, _ID_PICK, _ID_NORMAL, _ID_PUSH, _ID_TEXT = 0, 1, 2, 3, 4
_ID_LABELS = 100    # first id of the optional per-point index labels


def _resolve_npz_path(cli_path):
    # Same model as the executor, or the agreement check this script exists for
    # is meaningless.
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
    parser.add_argument("--idx", type=int, default=None,
                        help="index to show at startup; the node stays live either way")
    parser.add_argument("--npz-path", default="", dest="npz_path",
                        help="object model .npz (default: the executor's ~npz_path)")
    parser.add_argument("--frame", default="object_link",
                        help="frame to draw in; the link cloud is already expressed "
                             "in it. Use 'world' to inspect the model with no mocap")
    parser.add_argument("--topic", default="push_test_marker",
                        help="MarkerArray topic (rviz already displays the default)")
    parser.add_argument("--idx-topic", default="~loc_idx", dest="idx_topic",
                        help="std_msgs/Int32 topic to set the index at runtime")
    parser.add_argument("--click-topic", default="/clicked_point", dest="click_topic",
                        help="rviz 'Publish Point' topic; a click picks the nearest "
                             "model point. Empty string disables the reverse lookup")
    parser.add_argument("--normal-len", type=float, default=0.08, dest="normal_len",
                        help="normal / push arrow length (m)")
    parser.add_argument("--shrink", type=float, default=None,
                        help="contact-cloud margin (default: the executor's "
                             "~pcl_shrink); 1.0 draws the nominal model")
    parser.add_argument("--labels", action="store_true",
                        help="draw every point's index as text; unreadable at normal "
                             "zoom, so off by default")
    parser.add_argument("--no-prompt", action="store_true", dest="no_prompt",
                        help="skip the stdin prompt and just spin on the topics")
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def _marker(frame, marker_id, mtype, stamp):
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns = _NS
    m.id = marker_id
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def _arrow(frame, marker_id, stamp, tail, direction, length, color):
    a = _marker(frame, marker_id, Marker.ARROW, stamp)
    tip = np.asarray(tail) + length * np.asarray(direction)
    a.points = [Point(*[float(v) for v in tail]), Point(*[float(v) for v in tip])]
    a.scale.x, a.scale.y, a.scale.z = 0.006, 0.014, 0.02
    a.color = color
    return a


def _label_markers(frame, stamp, link_pts):
    # One text marker per point; ids start past the fixed markers so a re-pick
    # overwrites the picked-point markers without disturbing these.
    out = []
    for i, p in enumerate(link_pts):
        t = _marker(frame, _ID_LABELS + i, Marker.TEXT_VIEW_FACING, stamp)
        t.pose.position = Point(float(p[0]), float(p[1]), float(p[2]))
        t.scale.z = 0.008
        t.color = ColorRGBA(0.8, 0.8, 0.8, 0.7)
        t.text = str(i)
        out.append(t)
    return out


class Probe(object):
    """Holds the model and the marker publisher; every input path calls show()."""

    def __init__(self, config):
        self.config = config
        npz = _resolve_npz_path(config.npz_path)
        if not npz:
            rospy.logfatal("No object model: pass --npz-path or set "
                           "/executor_node/npz_path.")
            sys.exit(1)
        shrink = _resolve_shrink(config.shrink)
        pts, self.normals, scale, centroid = ol.load_model_npz(npz)
        self.link_pts = ol.link_frame_cloud(pts, scale, centroid, shrink)
        self.n_points = self.link_pts.shape[0]
        rospy.loginfo("object model: %s (%d points, scale=%.4f pcl_shrink=%.4f); "
                      "drawing in %s on %s", npz, self.n_points, scale, shrink,
                      config.frame, config.topic)

        # Topic and click callbacks fire on their own threads, so serialize the
        # build-and-publish rather than interleaving two marker arrays.
        self._lock = threading.Lock()
        self.pub = rospy.Publisher(config.topic, MarkerArray, queue_size=1, latch=True)
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf)

    def _build_markers(self, idx, point):
        stamp = rospy.Time.now()
        frame = self.config.frame

        cloud = _marker(frame, _ID_CLOUD, Marker.SPHERE_LIST, stamp)
        cloud.scale.x = cloud.scale.y = cloud.scale.z = 0.008
        cloud.color = ColorRGBA(0.6, 0.6, 0.6, 0.6)
        cloud.points = [Point(float(p[0]), float(p[1]), float(p[2]))
                        for p in self.link_pts]

        pick = _marker(frame, _ID_PICK, Marker.SPHERE, stamp)
        pick.pose.position = Point(*[float(v) for v in point])
        pick.scale.x = pick.scale.y = pick.scale.z = 0.025
        pick.color = ColorRGBA(1.0, 0.2, 0.2, 1.0)

        # +normal is the outward surface direction; the sim's cone
        # parametrization keeps a push at this index within 20 deg of -normal,
        # so -normal is the arrow to draw.
        n = np.asarray(self.normals[idx], dtype=np.float64)
        normal_arrow = _arrow(frame, _ID_NORMAL, stamp, point, n,
                              self.config.normal_len, ColorRGBA(0.1, 0.9, 0.9, 1.0))
        push_arrow = _arrow(frame, _ID_PUSH, stamp, point, -n,
                            self.config.normal_len, ColorRGBA(0.2, 0.6, 1.0, 1.0))

        text = _marker(frame, _ID_TEXT, Marker.TEXT_VIEW_FACING, stamp)
        text.pose.position = Point(float(point[0]), float(point[1]),
                                   float(point[2]) + 0.06)
        text.scale.z = 0.03
        text.color = ColorRGBA(1.0, 1.0, 0.1, 1.0)
        text.text = "idx %d (%.3f, %.3f, %.3f)" % (idx, point[0], point[1], point[2])

        arr = MarkerArray()
        arr.markers = [cloud, pick, normal_arrow, push_arrow, text]
        if self.config.labels:
            arr.markers.extend(_label_markers(frame, stamp, self.link_pts))
        return arr

    def show(self, idx):
        # A negative index is the executor's "use push_point instead" sentinel, which
        # this script has no point to fall back to; reject it before resolving.
        if idx < 0:
            rospy.logwarn("loc_idx=%d is the push_point sentinel, not an index", idx)
            return False
        # resolve_push_point is the executor's own lookup, so an index this script
        # accepts is one the executor accepts, and to the same point.
        try:
            point, _src = el.resolve_push_point(idx, None, self.link_pts)
        except ValueError as exc:
            rospy.logwarn("%s", exc)
            return False
        with self._lock:
            self.pub.publish(self._build_markers(idx, point))
        rospy.loginfo("loc_idx=%d -> (%.4f, %.4f, %.4f) in %s  |  --loc-idx %d  or  "
                      "--push-point %.4f %.4f %.4f",
                      idx, point[0], point[1], point[2], self.config.frame, idx,
                      point[0], point[1], point[2])
        return True

    def on_idx(self, msg):
        self.show(int(msg.data))

    def on_click(self, msg):
        # rviz publishes in its fixed frame, so bring the click into the cloud's
        # frame before the nearest-point search.
        try:
            local = self.buf.transform(msg, self.config.frame,
                                       timeout=rospy.Duration(0.5))
        except tf2_ros.TransformException as exc:
            rospy.logwarn("clicked point %s -> %s failed: %s",
                          msg.header.frame_id, self.config.frame, exc)
            return
        p = np.array([local.point.x, local.point.y, local.point.z])
        d = np.linalg.norm(self.link_pts - p, axis=1)
        idx = int(np.argmin(d))
        rospy.loginfo("click %.3f m from nearest model point", float(d[idx]))
        self.show(idx)


def _prompt_loop(probe):
    while not rospy.is_shutdown():
        try:
            raw = input("loc idx [0,%d) (q to quit): " % probe.n_points).strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return
        if raw.lower() in ("q", "quit", "exit"):
            return
        if not raw:
            continue
        try:
            idx = int(raw)
        except ValueError:
            rospy.logwarn("not an integer: %r", raw)
            continue
        probe.show(idx)


def main():
    config = _parse_args()
    rospy.init_node("debug_loc_idx_viz")
    probe = Probe(config)

    rospy.Subscriber(config.idx_topic, Int32, probe.on_idx, queue_size=1)
    rospy.loginfo("set the index live:  rostopic pub -1 %s std_msgs/Int32 \"data: 5\"",
                  rospy.resolve_name(config.idx_topic))
    if config.click_topic:
        rospy.Subscriber(config.click_topic, PointStamped, probe.on_click, queue_size=1)
        rospy.loginfo("or use rviz 'Publish Point' on the object (%s) for the "
                      "nearest index", config.click_topic)

    if config.idx is not None:
        probe.show(config.idx)

    # roslaunch gives nodes no stdin, so only prompt when there is a terminal.
    if config.no_prompt or not sys.stdin.isatty():
        rospy.spin()
        return
    _prompt_loop(probe)


if __name__ == "__main__":
    main()

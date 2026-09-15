#!/usr/bin/env python3
import copy
from collections import deque, namedtuple

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from npm_policy import obs_viz as ov

Snapshot = namedtuple("Snapshot", "stamp pose step success")

SNAPSHOT_COLOR = ColorRGBA(0.835, 0.369, 0.0, 1.0)
SUCCESS_COLOR = ColorRGBA(0.0, 0.620, 0.451, 1.0)


class SettleSnapshots(object):

    def __init__(self, mesh_path, frame="world", ns="settled_object",
                 max_snapshots=50, color=None, success_color=None,
                 alpha_min=0.15, alpha_max=0.85, scale=1.0, label=True,
                 label_size=0.05, label_z=0.12):
        self.mesh_path = str(mesh_path)
        self.frame = frame
        self.ns = ns
        self.label_ns = ns + "_labels"
        self.max_snapshots = int(max_snapshots)
        self.color = color if color is not None else SNAPSHOT_COLOR
        self.success_color = success_color if success_color is not None \
            else SUCCESS_COLOR
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.scale = float(scale)
        self.label = bool(label)
        self.label_size = float(label_size)
        self.label_z = float(label_z)
        self.snaps = deque(maxlen=self.max_snapshots
                           if self.max_snapshots > 0 else None)
        self.dropped = 0

    def clear(self):
        self.snaps.clear()
        self.dropped = 0

    def __len__(self):
        return len(self.snaps)

    def append(self, state, step, success=False):
        if state is None:
            return False
        if self.max_snapshots > 0 and len(self.snaps) == self.max_snapshots:
            self.dropped += 1
        self.snaps.append(Snapshot(state.header.stamp, copy.deepcopy(state.pose),
                                   int(step), bool(success)))
        return True

    def _alpha(self, i):
        n = len(self.snaps)
        if n < 2:
            return self.alpha_max
        return self.alpha_min + (self.alpha_max - self.alpha_min) * (i / float(n - 1))

    def _color(self, snap, i):
        base = self.success_color if snap.success else self.color
        return ColorRGBA(base.r, base.g, base.b, self._alpha(i))

    def marker_array(self, stamp):
        arr = MarkerArray()
        for i, snap in enumerate(self.snaps):
            m = ov.mesh_resource_marker(self.mesh_path, copy.deepcopy(snap.pose),
                                        stamp, frame=self.frame,
                                        color=self._color(snap, i), ns=self.ns,
                                        scale=self.scale)
            m.id = i
            m.lifetime = rospy.Duration(0.0)
            arr.markers.append(m)
            if self.label:
                arr.markers.append(self._label_marker(snap, i, stamp))
        return arr

    def _label_marker(self, snap, i, stamp):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.frame
        m.ns = self.label_ns
        m.id = i
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = Point(snap.pose.position.x, snap.pose.position.y,
                                snap.pose.position.z + self.label_z)
        m.pose.orientation.w = 1.0
        m.scale.z = self.label_size
        m.color = self._color(snap, i)
        m.lifetime = rospy.Duration(0.0)
        m.text = "#%d goal" % snap.step if snap.success else "#%d" % snap.step
        return m

    def delete_all_marker_array(self, stamp):
        arr = MarkerArray()
        for ns in (self.ns, self.label_ns):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.frame
            m.ns = ns
            m.id = 0
            m.action = Marker.DELETEALL
            arr.markers.append(m)
        return arr

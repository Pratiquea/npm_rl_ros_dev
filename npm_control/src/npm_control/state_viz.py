#!/usr/bin/env python3
import math
from collections import deque, namedtuple

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, Twist
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from trajectory_msgs.msg import (MultiDOFJointTrajectory,
                                 MultiDOFJointTrajectoryPoint)
from visualization_msgs.msg import Marker, MarkerArray


def _arrow(state, tip, marker_id, ns, color, shaft=0.02, head=0.04):
    m = Marker()
    m.header = state.header
    m.ns = ns
    m.id = marker_id
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.scale.x, m.scale.y, m.scale.z = shaft, head, 0.0
    m.pose.orientation.w = 1.0
    m.color = color
    m.lifetime = rospy.Duration(0.5)
    p = state.pose.position
    m.points = [Point(p.x, p.y, p.z), Point(*tip)]
    return m


def state_markers(state, vel_scale=1.0, dir_len=0.3):
    p = state.pose.position
    v = state.lin_vel
    vel_tip = (p.x + vel_scale * v.x, p.y + vel_scale * v.y, p.z + vel_scale * v.z)
    dir_tip = (p.x + dir_len * state.dir_to_goal_x,
               p.y + dir_len * state.dir_to_goal_y, p.z)

    arr = MarkerArray()
    arr.markers.append(_arrow(state, vel_tip, 0, "object_lin_vel",
                              ColorRGBA(0.2, 0.9, 0.2, 1.0)))
    arr.markers.append(_arrow(state, dir_tip, 1, "object_dir_to_goal",
                              ColorRGBA(0.9, 0.6, 0.1, 1.0)))
    return arr


TrailPoint = namedtuple("TrailPoint", "stamp pose twist")


def _copy_pose(pose):
    out = Pose()
    out.position.x, out.position.y, out.position.z = (pose.position.x,
                                                      pose.position.y,
                                                      pose.position.z)
    out.orientation.x, out.orientation.y = pose.orientation.x, pose.orientation.y
    out.orientation.z, out.orientation.w = pose.orientation.z, pose.orientation.w
    return out


def _twist_from_state(state):
    tw = Twist()
    tw.linear.x, tw.linear.y, tw.linear.z = (state.lin_vel.x, state.lin_vel.y,
                                             state.lin_vel.z)
    tw.angular.x, tw.angular.y, tw.angular.z = (state.ang_vel.x, state.ang_vel.y,
                                                state.ang_vel.z)
    return tw


TRAJ_COLOR = ColorRGBA(0.941, 0.894, 0.259, 1.0)


class TrajectoryTrail(object):

    def __init__(self, min_step_m=0.005, max_points=5000, frame="world",
                 color=None, width=0.008, ns="object_trajectory",
                 joint_name="object_link"):
        self.min_step = float(min_step_m)
        self.max_points = int(max_points)
        self.frame = frame
        self.color = color if color is not None else TRAJ_COLOR
        self.width = float(width)
        self.ns = ns
        self.joint_name = joint_name
        self.points = deque(maxlen=self.max_points if self.max_points > 0 else None)
        self.frozen = False
        self.dropped = 0

    def clear(self):
        self.points.clear()
        self.frozen = False
        self.dropped = 0

    def freeze(self):
        self.frozen = True

    def __len__(self):
        return len(self.points)

    def append(self, state):
        if self.frozen or state is None:
            return False
        p = state.pose.position
        if self.points:
            q = self.points[-1].pose.position
            step = math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2)
            if step < self.min_step:
                return False

        if self.max_points > 0 and len(self.points) == self.max_points:
            self.dropped += 1
        self.points.append(TrailPoint(state.header.stamp,
                                      _copy_pose(state.pose),
                                      _twist_from_state(state)))
        return True

    def path_msg(self, stamp):
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame
        for tp in self.points:
            ps = PoseStamped()
            ps.header.stamp = tp.stamp
            ps.header.frame_id = self.frame
            ps.pose = tp.pose
            msg.poses.append(ps)
        return msg

    def multidof_msg(self, stamp):
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame
        msg.joint_names = [self.joint_name]
        if not self.points:
            return msg
        t0 = self.points[0].stamp
        for tp in self.points:
            pt = MultiDOFJointTrajectoryPoint()
            tf = Transform()
            tf.translation.x = tp.pose.position.x
            tf.translation.y = tp.pose.position.y
            tf.translation.z = tp.pose.position.z
            tf.rotation = tp.pose.orientation
            pt.transforms = [tf]
            pt.velocities = [tp.twist]
            pt.time_from_start = tp.stamp - t0
            msg.points.append(pt)
        return msg

    def marker_msg(self, stamp):
        if len(self.points) < 2:
            return None
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.frame
        m.ns = self.ns
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = self.width
        m.pose.orientation.w = 1.0
        m.color = self.color
        m.lifetime = rospy.Duration(0.0)
        m.points = [Point(tp.pose.position.x, tp.pose.position.y,
                          tp.pose.position.z) for tp in self.points]
        return m

    def delete_marker(self, stamp):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.frame
        m.ns = self.ns
        m.id = 0
        m.action = Marker.DELETE
        return m

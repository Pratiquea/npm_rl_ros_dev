#!/usr/bin/env python3
"""
Rviz arrows for the filtered object state: velocity and direction to the goal.
"""
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
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
    """Velocity arrow (scaled) and unit direction-to-goal arrow, both in world.

    The direction arrow must point at the world origin from any object position;
    that is the check that dir_to_goal is right way round.
    """
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

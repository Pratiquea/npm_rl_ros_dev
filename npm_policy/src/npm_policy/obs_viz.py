#!/usr/bin/env python3
"""
Rviz message builders for the observation point cloud and its normals, shared
by policy_node and test_viz_obs_cloud so both draw the identical thing.
"""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker

from npm_policy import obs_lib as ol

# Render frame only: object position, IDENTITY rotation (see obs_frame_tf). NOT
# the calibrated link frame - that is object_link, owned by npm_calib. Pointing
# both at one name gave the frame two parents and made the extrinsic look
# non-rigid in rviz.
OBS_FRAME = "object_obs"

# Render frame for the LITERAL obs blocks, which are normalized about the COM and
# unscaled. Those points are not metric, so they get parked at a fixed origin
# (the goal by default) instead of on the object like OBS_FRAME.
RAW_OBS_FRAME = "object_obs_raw"


def obs_frame_tf(pos_world, stamp, parent="world", child=OBS_FRAME):
    """TF at the object's position with IDENTITY rotation.

    The observation cloud is rotated but not translated, so publishing it in
    this frame overlays it on the physical object: if the two disagree in rviz,
    the rotation is wrong. Giving the frame the object's rotation instead would
    cancel the very thing under test.
    """
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = float(pos_world[0])
    tf.transform.translation.y = float(pos_world[1])
    tf.transform.translation.z = float(pos_world[2])
    tf.transform.rotation.w = 1.0
    return tf


def cloud_msg(pc_flat, stamp, frame=OBS_FRAME):
    # pc_flat is the observation's pc block (N*3, row-major).
    pts = np.asarray(pc_flat, dtype=np.float32).reshape(-1, 3)
    header = Header(stamp=stamp, frame_id=frame)
    return point_cloud2.create_cloud_xyz32(header, pts.tolist())


def normals_marker(pc_flat, nrm_flat, stamp, frame=OBS_FRAME, length=0.05,
                   width=0.002, ns="obs_normals"):
    """One LINE_LIST marker: a segment per point along its rotated normal."""
    pts = np.asarray(pc_flat, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(nrm_flat, dtype=np.float64).reshape(-1, 3)

    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame
    m.ns = ns
    m.id = 0
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = width
    m.pose.orientation.w = 1.0
    m.color = ColorRGBA(0.1, 0.9, 0.9, 1.0)
    m.lifetime = rospy.Duration(0.5)
    for p, n in zip(pts, nrm):
        tip = p + length * n
        m.points.append(Point(float(p[0]), float(p[1]), float(p[2])))
        m.points.append(Point(float(tip[0]), float(tip[1]), float(tip[2])))
    return m


def reach_marker(pc_flat, reachable, stamp, frame=OBS_FRAME, size=0.012,
                 ns="reachable"):
    """POINTS marker colouring each obs point by the reachability gate.

    Green is reachable, grey is not, and the chosen point is what the argmax was
    allowed to pick from. A separate marker rather than rgb on the shared cloud:
    cloud_msg is create_cloud_xyz32 and every consumer of that topic, rviz
    configs included, expects xyz32.

    pc_flat must be the same rotated link cloud the obs cloud is drawn from, so
    the two overlay exactly in OBS_FRAME.
    """
    pts = np.asarray(pc_flat, dtype=np.float64).reshape(-1, 3)
    ok = np.asarray(reachable, dtype=bool).reshape(-1)

    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame
    m.ns = ns
    m.id = 0
    m.type = Marker.POINTS
    m.action = Marker.ADD
    m.scale.x = size
    m.scale.y = size
    m.pose.orientation.w = 1.0
    m.lifetime = rospy.Duration(0.5)
    green = ColorRGBA(0.1, 0.9, 0.2, 1.0)
    grey = ColorRGBA(0.5, 0.5, 0.5, 1.0)
    for p, good in zip(pts, ok):
        m.points.append(Point(float(p[0]), float(p[1]), float(p[2])))
        m.colors.append(green if good else grey)
    return m


def cloud_and_normals(pc_flat, nrm_flat, pos_world, stamp, parent="world",
                      length=0.05, child=OBS_FRAME, ns="obs_normals"):
    # The three messages always travel together; one call keeps their stamps equal.
    return (obs_frame_tf(pos_world, stamp, parent=parent, child=child),
            cloud_msg(pc_flat, stamp, frame=child),
            normals_marker(pc_flat, nrm_flat, stamp, frame=child, length=length,
                           ns=ns))


def raw_obs_cloud_and_normals(obs, origin, stamp, parent="world", length=0.05):
    """Draw the literal obs blocks the network consumes, obs[0:384] and obs[384:768].

    Not the link cloud: these points are normalized about the COM and unscaled, so
    they are parked at `origin` (the goal) rather than overlaid on the object. If
    this cloud ever matches the link cloud's size, the obs pipeline has regressed.
    """
    return cloud_and_normals(obs[ol.PC_OFF:ol.NORMALS_OFF],
                             obs[ol.NORMALS_OFF:ol.EXTRAS_OFF],
                             origin, stamp, parent=parent, length=length,
                             child=RAW_OBS_FRAME, ns="raw_obs_normals")

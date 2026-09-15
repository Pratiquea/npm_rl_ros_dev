#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker

from npm_policy import obs_lib as ol

OBS_FRAME = "object_obs"

RAW_OBS_FRAME = "object_obs_raw"


def obs_frame_tf(pos_world, stamp, parent="world", child=OBS_FRAME):
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
    pts = np.asarray(pc_flat, dtype=np.float32).reshape(-1, 3)
    header = Header(stamp=stamp, frame_id=frame)
    return point_cloud2.create_cloud_xyz32(header, pts.tolist())


def normals_marker(pc_flat, nrm_flat, stamp, frame=OBS_FRAME, length=0.05,
                   width=0.002, ns="obs_normals"):
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
    return (obs_frame_tf(pos_world, stamp, parent=parent, child=child),
            cloud_msg(pc_flat, stamp, frame=child),
            normals_marker(pc_flat, nrm_flat, stamp, frame=child, length=length,
                           ns=ns))


def raw_obs_cloud_and_normals(obs, origin, stamp, parent="world", length=0.05):
    return cloud_and_normals(obs[ol.PC_OFF:ol.NORMALS_OFF],
                             obs[ol.NORMALS_OFF:ol.EXTRAS_OFF],
                             origin, stamp, parent=parent, length=length,
                             child=RAW_OBS_FRAME, ns="raw_obs_normals")


MESH_COLOR = ColorRGBA(0.337, 0.706, 0.914, 0.70)


def _mesh_marker_base(pose, stamp, frame, color, ns):
    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame
    m.ns = ns
    m.id = 0
    m.action = Marker.ADD
    m.pose = pose
    m.color = color if color is not None else MESH_COLOR
    m.lifetime = rospy.Duration(0.5)
    return m


def mesh_resource_marker(mesh_path, pose, stamp, frame="world", color=None,
                         ns="object_mesh", scale=1.0):
    m = _mesh_marker_base(pose, stamp, frame, color, ns)
    m.type = Marker.MESH_RESOURCE
    m.mesh_resource = "file://" + str(mesh_path)
    m.mesh_use_embedded_materials = False
    m.scale.x = m.scale.y = m.scale.z = float(scale)
    return m


def mesh_triangles_marker(verts, tris, pose, stamp, frame="world", color=None,
                          ns="object_mesh", scale=1.0):
    m = _mesh_marker_base(pose, stamp, frame, color, ns)
    m.type = Marker.TRIANGLE_LIST
    m.scale.x = m.scale.y = m.scale.z = float(scale)
    v = np.asarray(verts, dtype=np.float64)
    for tri in np.asarray(tris, dtype=np.int64).reshape(-1, 3):
        for idx in tri:
            p = v[idx]
            m.points.append(Point(float(p[0]), float(p[1]), float(p[2])))
    return m


def mesh_delete_marker(stamp, frame="world", ns="object_mesh"):
    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame
    m.ns = ns
    m.id = 0
    m.action = Marker.DELETE
    return m

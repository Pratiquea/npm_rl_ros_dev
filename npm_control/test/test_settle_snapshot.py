#!/usr/bin/env python3
import os
import sys

import rospy
from npm_msgs.msg import ObjectState
from visualization_msgs.msg import Marker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from npm_control.settle_snapshot import (SettleSnapshots, SNAPSHOT_COLOR,
                                         SUCCESS_COLOR)

MESH = "/tmp/Paralelopiped.obj"


def _state(x, y=0.0, z=0.0, t=0.0, quat=(0.0, 0.0, 0.0, 1.0)):
    m = ObjectState()
    m.header.stamp = rospy.Time(t)
    m.header.frame_id = "world"
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    (m.pose.orientation.x, m.pose.orientation.y,
     m.pose.orientation.z, m.pose.orientation.w) = quat
    return m


def _meshes(arr):
    return [m for m in arr.markers if m.type == Marker.MESH_RESOURCE]


def _labels(arr):
    return [m for m in arr.markers if m.type == Marker.TEXT_VIEW_FACING]


def test_one_mesh_per_settle():
    snap = SettleSnapshots(MESH)
    for i in range(4):
        assert snap.append(_state(0.1 * i), i) is True
    arr = snap.marker_array(rospy.Time(0))
    meshes = _meshes(arr)
    assert len(snap) == len(meshes) == 4, len(meshes)
    for m in meshes:
        assert m.mesh_resource == "file://" + MESH, m.mesh_resource
        assert m.header.frame_id == "world"
        assert m.action == Marker.ADD
        assert m.lifetime == rospy.Duration(0.0)
    print("PASS one persistent MESH_RESOURCE marker per settle, in `world`")


def test_no_motion_gate():
    snap = SettleSnapshots(MESH)
    for i in range(5):
        assert snap.append(_state(0.0), i) is True
    assert len(snap) == 5
    print("PASS a settle is stored even when the object did not move")


def test_none_state_is_ignored():
    snap = SettleSnapshots(MESH)
    assert snap.append(None, 0) is False
    assert len(snap) == 0


def test_pose_is_copied_not_aliased():
    snap = SettleSnapshots(MESH)
    state = _state(1.0)
    snap.append(state, 0)
    state.pose.position.x = 99.0
    assert snap.snaps[0].pose.position.x == 1.0
    arr = snap.marker_array(rospy.Time(0))
    _meshes(arr)[0].pose.position.x = 42.0
    assert snap.snaps[0].pose.position.x == 1.0
    print("PASS the store copies the pose, and the markers copy it again")


def test_orientation_survives_as_ros_xyzw():
    quat = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    snap = SettleSnapshots(MESH)
    snap.append(_state(0.3, quat=quat), 1)
    q = _meshes(snap.marker_array(rospy.Time(0)))[0].pose.orientation
    assert (q.x, q.y, q.z, q.w) == quat, (q.x, q.y, q.z, q.w)
    print("PASS the settled orientation reaches the marker in ROS xyzw")


def test_cap_drops_oldest():
    snap = SettleSnapshots(MESH, max_snapshots=10)
    for i in range(50):
        snap.append(_state(0.01 * i), i)
    assert len(snap) == 10, len(snap)
    assert snap.dropped == 40, snap.dropped
    assert snap.snaps[-1].step == 49
    print("PASS a full memory drops its oldest snapshot and stays capped")


def test_ids_are_positions_so_dropped_meshes_cannot_linger():
    snap = SettleSnapshots(MESH, max_snapshots=3, label=False)
    for i in range(9):
        snap.append(_state(0.01 * i), i)
    ids = [m.id for m in _meshes(snap.marker_array(rospy.Time(0)))]
    assert ids == [0, 1, 2], ids
    print("PASS marker ids are store positions, never macro-step numbers")


def test_age_ramp_runs_faint_to_solid():
    snap = SettleSnapshots(MESH, alpha_min=0.2, alpha_max=0.8)
    for i in range(5):
        snap.append(_state(0.01 * i), i)
    alphas = [m.color.a for m in _meshes(snap.marker_array(rospy.Time(0)))]
    assert abs(alphas[0] - 0.2) < 1e-9 and abs(alphas[-1] - 0.8) < 1e-9, alphas
    assert all(b > a for a, b in zip(alphas, alphas[1:])), alphas
    print("PASS the oldest snapshot is the faintest and the newest is solid")


def test_single_snapshot_is_fully_opaque():
    snap = SettleSnapshots(MESH, alpha_min=0.15, alpha_max=0.85)
    snap.append(_state(0.0), 0)
    assert abs(_meshes(snap.marker_array(rospy.Time(0)))[0].color.a - 0.85) < 1e-9


def test_success_snapshot_is_the_only_green_one():
    snap = SettleSnapshots(MESH)
    snap.append(_state(1.0), 0)
    snap.append(_state(0.1), 1, success=True)
    meshes = _meshes(snap.marker_array(rospy.Time(0)))
    assert (meshes[0].color.r, meshes[0].color.g) == (SNAPSHOT_COLOR.r,
                                                      SNAPSHOT_COLOR.g)
    assert (meshes[1].color.r, meshes[1].color.g) == (SUCCESS_COLOR.r,
                                                      SUCCESS_COLOR.g)
    print("PASS the settle that reached the circle is the green one")


def test_labels_carry_the_macro_step_and_own_namespace():
    snap = SettleSnapshots(MESH, ns="settled_object")
    snap.append(_state(0.0, z=0.5), 3)
    snap.append(_state(0.1, z=0.5), 4, success=True)
    arr = snap.marker_array(rospy.Time(0))
    labels = _labels(arr)
    assert [m.text for m in labels] == ["#3", "#4 goal"], [m.text for m in labels]
    assert all(m.ns == "settled_object_labels" for m in labels)
    assert all(m.ns == "settled_object" for m in _meshes(arr))
    assert labels[0].pose.position.z > 0.5
    print("PASS labels number the macro-steps and sit in their own namespace")


def test_labels_can_be_switched_off():
    snap = SettleSnapshots(MESH, label=False)
    snap.append(_state(0.0), 0)
    assert _labels(snap.marker_array(rospy.Time(0))) == []


def test_clear_deletes_both_namespaces():
    snap = SettleSnapshots(MESH, ns="settled_object")
    for i in range(3):
        snap.append(_state(0.01 * i), i)
    arr = snap.delete_all_marker_array(rospy.Time(0))
    assert [m.action for m in arr.markers] == [Marker.DELETEALL] * 2
    assert sorted(m.ns for m in arr.markers) == ["settled_object",
                                                 "settled_object_labels"]
    snap.clear()
    assert len(snap) == 0 and snap.dropped == 0
    assert snap.marker_array(rospy.Time(0)).markers == []
    print("PASS clear deletes the meshes and the labels together")


def main():
    test_one_mesh_per_settle()
    test_no_motion_gate()
    test_none_state_is_ignored()
    test_pose_is_copied_not_aliased()
    test_orientation_survives_as_ros_xyzw()
    test_cap_drops_oldest()
    test_ids_are_positions_so_dropped_meshes_cannot_linger()
    test_age_ramp_runs_faint_to_solid()
    test_single_snapshot_is_fully_opaque()
    test_success_snapshot_is_the_only_green_one()
    test_labels_carry_the_macro_step_and_own_namespace()
    test_labels_can_be_switched_off()
    test_clear_deletes_both_namespaces()
    print("ALL PASS")


if __name__ == "__main__":
    main()

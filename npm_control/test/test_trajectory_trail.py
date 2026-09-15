#!/usr/bin/env python3
import os
import sys

import rospy
from npm_msgs.msg import ObjectState

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from npm_control.state_viz import TrajectoryTrail, TRAJ_COLOR

MIN_STEP = 0.005


def _state(x, y=0.0, z=0.0, t=0.0, vel=0.0):
    m = ObjectState()
    m.header.stamp = rospy.Time(t)
    m.header.frame_id = "world"
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    m.pose.orientation.w = 1.0
    m.lin_vel.x, m.lin_vel.y, m.lin_vel.z = vel, 2 * vel, 3 * vel
    m.ang_vel.x, m.ang_vel.y, m.ang_vel.z = 4 * vel, 5 * vel, 6 * vel
    return m


def test_first_sample_is_kept():
    trail = TrajectoryTrail(min_step_m=MIN_STEP)
    assert trail.append(_state(0.0)) is True
    assert len(trail) == 1


def test_below_min_step_is_dropped():
    trail = TrajectoryTrail(min_step_m=MIN_STEP)
    trail.append(_state(0.0))
    for i in range(100):
        assert trail.append(_state(0.001 * (i % 3))) is False
    assert len(trail) == 1
    print("PASS sub-min_step samples add no points")


def test_min_step_accumulates_not_resets():
    trail = TrajectoryTrail(min_step_m=MIN_STEP)
    trail.append(_state(0.0))
    kept = sum(trail.append(_state(0.001 * (i + 1))) for i in range(20))
    assert 3 <= kept <= 4, kept
    xs = [tp.pose.position.x for tp in trail.points]
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    assert all(g >= MIN_STEP for g in gaps), gaps
    print("PASS drift below min_step still lands points once it accumulates")


def test_none_state_is_ignored():
    trail = TrajectoryTrail(min_step_m=MIN_STEP)
    assert trail.append(None) is False
    assert len(trail) == 0


def test_cap_drops_oldest():
    trail = TrajectoryTrail(min_step_m=MIN_STEP, max_points=10)
    for i in range(50):
        trail.append(_state(0.01 * i))
    assert len(trail) == 10, len(trail)
    assert trail.dropped == 40, trail.dropped
    assert abs(trail.points[-1].pose.position.x - 0.49) < 1e-9
    print("PASS a full trail drops its oldest point and stays capped")


def test_freeze_stops_appending_and_keeps_points():
    trail = TrajectoryTrail(min_step_m=MIN_STEP)
    for i in range(5):
        trail.append(_state(0.01 * i))
    trail.freeze()
    assert trail.append(_state(10.0)) is False
    assert len(trail) == 5
    assert trail.marker_msg(rospy.Time(0)) is not None
    print("PASS freeze stops the trail without erasing it")


def test_all_renderings_share_one_store():
    trail = TrajectoryTrail(min_step_m=MIN_STEP)
    for i in range(7):
        trail.append(_state(0.01 * i, 0.02 * i, t=i))
    path = trail.path_msg(rospy.Time(0))
    marker = trail.marker_msg(rospy.Time(0))
    multidof = trail.multidof_msg(rospy.Time(0))
    assert len(path.poses) == len(marker.points) == len(multidof.points) == 7
    for ps, pt, mp in zip(path.poses, marker.points, multidof.points):
        tr = mp.transforms[0].translation
        assert (ps.pose.position.x, ps.pose.position.y) == (pt.x, pt.y)
        assert (tr.x, tr.y, tr.z) == (ps.pose.position.x, ps.pose.position.y,
                                      ps.pose.position.z)
    assert path.header.frame_id == marker.header.frame_id == "world"
    assert multidof.header.frame_id == "world"
    assert marker.color == TRAJ_COLOR
    print("PASS Path, LINE_STRIP and MultiDOF carry the same points in `world`")


def test_multidof_carries_pose_and_twist_together():
    trail = TrajectoryTrail(min_step_m=MIN_STEP, joint_name="object_link")
    for i in range(4):
        trail.append(_state(0.1 * i, t=i, vel=0.5 * i))
    msg = trail.multidof_msg(rospy.Time(0))
    assert msg.joint_names == ["object_link"], msg.joint_names
    for i, pt in enumerate(msg.points):
        assert len(pt.transforms) == len(pt.velocities) == 1
        assert abs(pt.transforms[0].translation.x - 0.1 * i) < 1e-9
        rot = pt.transforms[0].rotation
        assert (rot.x, rot.y, rot.z, rot.w) == (0.0, 0.0, 0.0, 1.0)
        tw = pt.velocities[0]
        assert (tw.linear.x, tw.linear.y, tw.linear.z) == (0.5 * i, 1.0 * i, 1.5 * i)
        assert (tw.angular.x, tw.angular.y,
                tw.angular.z) == (2.0 * i, 2.5 * i, 3.0 * i)
    print("PASS MultiDOF points carry pose (xyzw) and twist together")


def test_multidof_time_from_start_runs_from_the_first_kept_point():
    trail = TrajectoryTrail(min_step_m=MIN_STEP, max_points=3)
    for i in range(6):
        trail.append(_state(0.1 * i, t=10.0 + i))
    msg = trail.multidof_msg(rospy.Time(0))
    offsets = [pt.time_from_start.to_sec() for pt in msg.points]
    assert offsets == [0.0, 1.0, 2.0], offsets
    print("PASS time_from_start is measured from the first kept point")


def test_multidof_empty_trail_is_still_valid():
    msg = TrajectoryTrail().multidof_msg(rospy.Time(0))
    assert msg.points == [] and msg.joint_names == ["object_link"]


def test_marker_needs_two_points():
    trail = TrajectoryTrail(min_step_m=MIN_STEP)
    assert trail.marker_msg(rospy.Time(0)) is None
    trail.append(_state(0.0))
    assert trail.marker_msg(rospy.Time(0)) is None
    trail.append(_state(0.5))
    assert trail.marker_msg(rospy.Time(0)) is not None


def test_pose_is_copied_not_aliased():
    trail = TrajectoryTrail(min_step_m=MIN_STEP)
    state = _state(1.0, vel=1.0)
    trail.append(state)
    state.pose.position.x = 99.0
    state.lin_vel.x = 99.0
    assert trail.points[0].pose.position.x == 1.0
    assert trail.points[0].twist.linear.x == 1.0
    print("PASS the trail copies the pose and twist instead of aliasing the message")


def test_clear_resets_everything():
    trail = TrajectoryTrail(min_step_m=MIN_STEP, max_points=3)
    for i in range(10):
        trail.append(_state(0.01 * i))
    trail.freeze()
    trail.clear()
    assert len(trail) == 0 and trail.dropped == 0 and trail.frozen is False
    assert trail.append(_state(0.0)) is True
    assert trail.delete_marker(rospy.Time(0)).action == 2


def main():
    test_first_sample_is_kept()
    test_below_min_step_is_dropped()
    test_min_step_accumulates_not_resets()
    test_none_state_is_ignored()
    test_cap_drops_oldest()
    test_freeze_stops_appending_and_keeps_points()
    test_all_renderings_share_one_store()
    test_multidof_carries_pose_and_twist_together()
    test_multidof_time_from_start_runs_from_the_first_kept_point()
    test_multidof_empty_trail_is_still_valid()
    test_marker_needs_two_points()
    test_pose_is_copied_not_aliased()
    test_clear_resets_everything()
    print("ALL PASS")


if __name__ == "__main__":
    main()

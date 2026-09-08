#!/usr/bin/env python3
"""Offline test for executor_lib: frame math + hybrid command structure."""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from npm_control import executor_lib as el


def main():
    # --- task frame: x-axis must equal the push direction ---
    force = np.array([-27.446, 74.843, 29.498])   # from the live policy smoke
    mag, q = el.task_frame_from_force(force)
    assert abs(mag - np.linalg.norm(force)) < 1e-6
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    x_axis = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(x_axis, force / np.linalg.norm(force), atol=1e-6), x_axis
    assert abs(np.linalg.det(R) - 1.0) < 1e-6   # proper rotation

    # --- hybrid command structure ---
    from bosdyn.api import arm_command_pb2
    Req = arm_command_pb2.ArmCartesianCommand.Request
    contact = [1.482, 0.757, 0.653]
    cmd, m, qq = el.build_arm_cartesian_command(contact, force, root_frame="odom",
                                                duration_s=1.5)
    req = cmd.synchronized_command.arm_command.arm_cartesian_command
    assert req.root_frame_name == "odom"
    assert req.x_axis == Req.AXIS_MODE_FORCE
    assert req.y_axis == Req.AXIS_MODE_POSITION
    assert req.z_axis == Req.AXIS_MODE_POSITION
    # roll about push axis is gauge-free -> compliant (FORCE), not servoed
    assert req.rx_axis == Req.AXIS_MODE_FORCE
    assert req.ry_axis == Req.AXIS_MODE_POSITION
    assert req.rz_axis == Req.AXIS_MODE_POSITION
    assert abs(req.wrench_trajectory_in_task.points[0].wrench.force.x - m) < 1e-5
    assert m <= 75.0 + 1e-6
    p = req.root_tform_task.position
    assert np.allclose([p.x, p.y, p.z], contact, atol=1e-6)
    # serializes (structurally valid protobuf)
    assert len(cmd.SerializeToString()) > 0

    # --- force clip enforced ---
    big = np.array([1000.0, 0.0, 0.0])
    _, mbig, _ = el.build_arm_cartesian_command([0, 0, 0], big)
    assert abs(mbig - 75.0) < 1e-6, mbig

    # --- tilted/azimuth push: task x-axis still == push dir, frame valid ---
    tilt = np.array([12.0, -40.0, 55.0])   # large azimuth + z component
    _, qt = el.task_frame_from_force(tilt)
    w, x, y, z = qt
    Rt = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(Rt @ np.array([1.0, 0.0, 0.0]), tilt / np.linalg.norm(tilt), atol=1e-6)
    assert abs(np.linalg.det(Rt) - 1.0) < 1e-6

    print("PASS: task x-axis==push dir, hybrid FORCE/POSITION (rx compliant), root=odom, |f|<=75, serializes")

    # --- follow_arm body offset threads into the FollowArmCommand ---
    from bosdyn.api import geometry_pb2
    off = geometry_pb2.Vec3(x=-0.7, y=0.0, z=0.0)
    cmd_off, _, _ = el.build_arm_cartesian_command(
        contact, force, follow_arm=True, body_offset_from_hand=off)
    mob = cmd_off.synchronized_command.mobility_command
    assert mob.HasField("follow_arm_request")
    bo = mob.follow_arm_request.body_offset_from_hand
    assert abs(bo.x - (-0.7)) < 1e-6, bo
    # follow_arm without an offset still builds (default follow), no offset set
    cmd_noff, _, _ = el.build_arm_cartesian_command(contact, force, follow_arm=True)
    assert cmd_noff.synchronized_command.mobility_command.HasField("follow_arm_request")
    # standoff builder threads the same offset
    scmd, _ = el.build_standoff_pose_command(
        contact, force, [1.0, 0.0, 0.0, 0.0], follow_arm=True,
        body_offset_from_hand=off)
    sbo = scmd.synchronized_command.mobility_command.follow_arm_request.body_offset_from_hand
    assert abs(sbo.x - (-0.7)) < 1e-6, sbo
    print("PASS: body_offset_from_hand threads through arm+standoff follow_arm commands")

    # --- manipulability: positive at a generic pose, collapses near full extension ---
    class _JS(object):
        def __init__(self, name, val):
            self.name = name
            self.position = type("P", (), {"value": val})()

    generic = [_JS(n, a) for n, a in zip(
        el._ARM_JOINT_NAMES, [0.0, -0.9, 1.6, 0.0, 0.7, 0.0])]
    w_generic = el.arm_manipulability(generic)
    assert w_generic > 1e-6, w_generic
    # arm stretched straight out (sh1/el0 ~ 0) is closer to a boundary singularity
    stretched = [_JS(n, a) for n, a in zip(
        el._ARM_JOINT_NAMES, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    w_stretched = el.arm_manipulability(stretched)
    assert w_stretched < w_generic, (w_stretched, w_generic)
    # missing arm joints -> nan (caller must guard)
    assert np.isnan(el.arm_manipulability([_JS("arm0.sh0", 0.0)]))
    print("PASS: arm_manipulability positive, drops toward extension, nan when joints missing")

    # --- retract command: eps=0 targets the point itself, every axis servoed ---
    # The retreat reuses the standoff builder with eps=0 so the commanded point IS the
    # clamped tip target. Position on all six axes is what gives the arm authority to
    # pull off the object; a FORCE axis along the push dir leaves it compliant on
    # exactly the axis it needs to move.
    tgt = [0.8, -0.3, 0.45]
    rcmd, standoff = el.build_standoff_pose_command(
        tgt, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], root_frame="odom", eps=0.0)
    assert np.allclose(standoff, tgt, atol=1e-9), standoff
    rreq = rcmd.synchronized_command.arm_command.arm_cartesian_command
    for ax in (rreq.x_axis, rreq.y_axis, rreq.z_axis,
               rreq.rx_axis, rreq.ry_axis, rreq.rz_axis):
        assert ax == Req.AXIS_MODE_POSITION, ax
    assert not rreq.HasField("wrench_trajectory_in_task")
    # base sub-commands: stand plants the base, SE2 drives it to an absolute goal
    assert el.build_stand_mobility_command().HasField("stand_request")
    assert el.build_se2_mobility_command(1.0, 2.0, 0.5).HasField("se2_trajectory_request")
    # ...and the SE2 goal carries NO end_time until robot_command(end_time_secs=...)
    # fills it -- an unset end_time lands already expired and the base never steps.
    assert not el.build_se2_mobility_command(
        1.0, 2.0, 0.5).se2_trajectory_request.HasField("end_time")
    print("PASS: retract command is all-POSITION at eps=0; stand/SE2 mobility sub-commands")

    # --- body clamp keeps the retract target out of the robot ---
    from bosdyn.api import geometry_pb2 as g
    from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME as FLAT

    def _snap(body_x, body_y, yaw):
        # odom -> flat_body at (body_x, body_y, 0) with the given yaw
        q = g.Quaternion(w=np.cos(yaw / 2.0), x=0.0, y=0.0, z=np.sin(yaw / 2.0))
        odom_T_body = g.SE3Pose(position=g.Vec3(x=body_x, y=body_y, z=0.0), rotation=q)
        return g.FrameTreeSnapshot(child_to_parent_edge_map={
            "odom": g.FrameTreeSnapshot.ParentEdge(parent_frame_name=""),
            FLAT: g.FrameTreeSnapshot.ParentEdge(
                parent_frame_name="odom", parent_tform_child=odom_T_body)})

    min_x = 0.55
    # body at origin facing +x: a target 0.90 m ahead is clear of the clamp
    snap = _snap(0.0, 0.0, 0.0)
    out, binds = el.clamp_target_to_body(snap, "odom", [0.90, 0.0, 0.3], min_x)
    assert binds is False and np.allclose(out, [0.90, 0.0, 0.3], atol=1e-9), (out, binds)
    # a target 0.20 m ahead would be inside the body -> pushed back out to min_x,
    # lateral and vertical components untouched
    out, binds = el.clamp_target_to_body(snap, "odom", [0.20, 0.10, 0.3], min_x)
    assert binds is True, binds
    assert np.allclose(out, [min_x, 0.10, 0.3], atol=1e-6), out
    # clamp is body-relative, not odom-relative: same geometry, base moved and yawed
    yaw = 2.0
    snap = _snap(-1.3, 0.7, yaw)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    base = np.array([-1.3, 0.7, 0.0])
    out, binds = el.clamp_target_to_body(
        snap, "odom", base + 0.20 * fwd + np.array([0, 0, 0.3]), min_x)
    assert binds is True, binds
    assert np.allclose(out, base + min_x * fwd + np.array([0, 0, 0.3]), atol=1e-6), out
    print("PASS: body clamp bounds the tip target at min_hand_body_x in the body frame")

    # --- reach gauge: radial from the SHOULDER pivot, not body-frame x ---
    from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME

    def _hand_snap(hand_pos_body):
        # body -> hand at hand_pos_body, no rotation
        return g.FrameTreeSnapshot(child_to_parent_edge_map={
            BODY_FRAME_NAME: g.FrameTreeSnapshot.ParentEdge(parent_frame_name=""),
            HAND_FRAME_NAME: g.FrameTreeSnapshot.ParentEdge(
                parent_frame_name=BODY_FRAME_NAME,
                parent_tform_child=g.SE3Pose(
                    position=g.Vec3(x=float(hand_pos_body[0]),
                                    y=float(hand_pos_body[1]),
                                    z=float(hand_pos_body[2])),
                    rotation=g.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)))})

    sh = el._SPOT_SHOULDER_FROM_BODY
    tip = np.zeros(3)                       # tip AT the hand frame, for this test
    # a) tip 0.99 m straight out in front of the shoulder -> reach 0.99, while the old
    #    body-x metric would have read 0.99 + the shoulder's own 0.292 m offset
    ahead = sh + np.array([0.99, 0.0, 0.0])
    assert abs(el.tip_reach_from_shoulder(_hand_snap(ahead), tip) - 0.99) < 1e-6
    assert abs(ahead[0] - 1.282) < 1e-3, ahead      # the old (wrong) reading
    # b) same extension pointed straight DOWN is the same reach, but its body x is the
    #    shoulder's offset alone: the old metric never fires on a low push
    down = sh + np.array([0.0, 0.0, -0.99])
    assert abs(el.tip_reach_from_shoulder(_hand_snap(down), tip) - 0.99) < 1e-6
    assert abs(down[0] - 0.292) < 1e-9, down
    # c) the tip offset from the hand frame is included
    off = np.array([0.12, 0.0, 0.02])
    r = el.tip_reach_from_shoulder(_hand_snap(ahead), off)
    assert abs(r - np.linalg.norm(np.array([0.99, 0.0, 0.0]) + off)) < 1e-6, r
    # d) the geometric maximum for the arm + 0.24/0.03 pusher tip
    reach_max = (el._SPOT_ARM_POE[2][0][0] + np.linalg.norm(el._SPOT_ARM_POE[3][0])
                 + np.hypot(0.24, 0.03))
    assert abs(reach_max - 0.9906) < 1e-3, reach_max
    print("PASS: reach gauge is radial from the shoulder pivot, tip offset included, "
          "geometric max %.4f m" % reach_max)

    # --- topple assist: tilt gauge, body-locked arm, base velocity mobility ---
    def R_axis(axis, deg):
        return Rotation.from_rotvec(np.deg2rad(deg) * np.asarray(axis, float)).as_matrix()

    assert abs(el.tilt_since(np.eye(3), np.eye(3))) < 1e-9
    # The gauge is a delta from the push-start pose, so the resting pose is arbitrary.
    # Frames whose +z points down or sideways broke the old absolute angle-from-upright
    # gauge: down flipped the sign of the gain, sideways drowned it in yaw.
    for ref in (np.eye(3), np.diag([1.0, -1.0, -1.0]), R_axis([1, 0, 0], 90.0)):
        assert abs(el.tilt_since(ref, ref)) < 1e-9
        for tip_axis in ([1, 0, 0], [0, 1, 0], [0.6, 0.8, 0.0]):
            # tips are world-frame rotations applied to the resting pose
            tipped = R_axis(tip_axis, 20.0) @ ref
            assert abs(np.rad2deg(el.tilt_since(ref, tipped)) - 20.0) < 1e-6
        # yaw is not tilt: spinning in place must not arm the topple assist
        for yaw_deg in (30.0, 150.0, -95.0):
            assert abs(el.tilt_since(ref, R_axis([0, 0, 1], yaw_deg) @ ref)) < 1e-9
        # yaw mixed into a tip still reports only the tip
        mixed = R_axis([0, 0, 1], 40.0) @ R_axis([1, 0, 0], 25.0) @ ref
        assert abs(np.rad2deg(el.tilt_since(ref, mixed)) - 25.0) < 1e-6
    # a full flip reads as pi, not as 0: the gauge must not wrap
    assert abs(el.tilt_since(np.eye(3), np.diag([1.0, -1.0, -1.0])) - np.pi) < 1e-9

    # --- topple duration override: tilt must still be rising to hold the clock open ---
    win = 0.5
    tipping = [(t_s, np.deg2rad(20.0 * t_s)) for t_s in np.arange(0.0, 1.01, 0.025)]
    # a partial window is not evidence either way: the override must not fire on it
    assert el.tilt_rise(tipping[:5], win) == (0.0, False)
    assert el.tilt_rise([], win) == (0.0, False)
    rise, valid = el.tilt_rise(tipping[-21:], win)      # exactly one window of samples
    assert valid and abs(np.rad2deg(rise) - 10.0) < 1e-6, rise
    # a tip that has finished or wedged reads ~0 and hands the clock back
    settled = [(t_s, np.deg2rad(25.0)) for t_s in np.arange(0.0, 1.01, 0.025)]
    rise, valid = el.tilt_rise(settled, win)
    assert valid and abs(rise) < 1e-12, rise
    # the window is the caller's: a longer history is judged on its whole span
    rise, valid = el.tilt_rise(tipping, win)
    assert valid and abs(np.rad2deg(rise) - 20.0) < 1e-6, rise

    mob = el.build_velocity_mobility_command(0.15, max_lin_vel=0.15)
    assert mob.HasField("se2_velocity_request"), mob
    vel = mob.se2_velocity_request.velocity
    assert abs(vel.linear.x - 0.15) < 1e-9 and abs(vel.linear.y) < 1e-9
    assert abs(vel.angular) < 1e-9    # the crawl is straight ahead, no yaw

    tool = (0.24, 0.03)
    cmd = el.build_body_locked_arm_command([0.85, 0.02, 0.45], [1.0, 0.0, 0.0, 0.0],
                                           duration_s=0.6, tool_offset=tool,
                                           mobility_command=mob)
    req = cmd.synchronized_command.arm_command.arm_cartesian_command
    # rooted in the BODY, not odom: that is what makes the arm ride the base
    assert req.root_frame_name == "flat_body", req.root_frame_name
    for axis in (req.x_axis, req.y_axis, req.z_axis,
                 req.rx_axis, req.ry_axis, req.rz_axis):
        assert axis == Req.AXIS_MODE_POSITION, axis   # rigid strut, no compliance
    assert not req.HasField("wrench_trajectory_in_task")
    assert req.force_remain_near_current_joint_configuration is True
    assert abs(req.wrist_tform_tool.position.x - tool[0]) < 1e-9
    assert abs(req.wrist_tform_tool.position.z - tool[1]) < 1e-9
    assert abs(req.root_tform_task.position.x - 0.85) < 1e-9
    # the base velocity must survive being bundled with the arm command
    assert cmd.synchronized_command.mobility_command.HasField("se2_velocity_request")
    print("PASS: tilt gauge is a yaw-free delta from the push-start pose, unwrapped; "
          "tilt rise needs a full window; "
          "body-locked arm is all-POSITION in flat_body "
          "and carries the base velocity")

    # --- contact latch: one bad force sample must not end a push ---
    dt = 1.0 / 40.0     # loop_rate
    def run(latch, t0, samples):
        # feed (force, count) pairs at loop rate; return the wall time after the last
        t = t0
        for force, n in samples:
            for _ in range(n):
                latch.update(force, t)
                t += dt
        return t

    made_n, eps, grace = 5.0, 3.0, 0.4
    lat = el.ContactLatch(made_n, eps, grace)
    t = run(lat, 0.0, [(0.0, 20)])
    assert not lat.made and not lat.lost      # approach zeros are not a loss
    t = run(lat, t, [(20.0, 10), (0.0, 8)])   # contact, then a 0.2 s dip
    assert lat.made and not lat.lost and lat.loaded
    t = run(lat, t, [(20.0, 5), (0.0, 8)])    # re-load resets the window
    assert not lat.lost and lat.low_for < grace
    t = run(lat, t, [(20.0, 5), (0.0, 24)])   # 0.6 s below eps ends it
    assert lat.lost and not lat.loaded

    # the hysteresis band [eps, made_n) still counts as contact
    lat = el.ContactLatch(made_n, eps, grace)
    t = run(lat, 0.0, [(20.0, 5), (4.0, 40)])
    assert lat.made and not lat.lost

    # a missing force estimate neither advances nor resets the window
    lat = el.ContactLatch(made_n, eps, grace)
    t = run(lat, 0.0, [(20.0, 5), (0.0, 8)])
    low = lat.low_for
    lat.update(None, t + 10.0)
    assert not lat.lost and abs(lat.low_for - low) < 1e-9

    # The latch _body_assist installs after RECONTACT. A fresh one cannot arm on an
    # object that has settled onto a stationary tool: the load sits in the [eps,
    # made_n) band, which reads as contact only to a latch that is ALREADY made.
    # This is what killed pushes 1 and 2 of bag spot_push_policy_10, as
    # end_reason="contact_lost" inside recenter.
    lat = el.ContactLatch(made_n, eps, grace)
    run(lat, 0.0, [(4.0, 40)])                # 1.0 s of settled load, never armed
    assert not lat.made and not lat.loaded
    lat = el.ContactLatch(made_n, eps, grace)
    assert lat.force_made(0.0)                # ...so RECONTACT arms it by fiat
    t = run(lat, 0.0, [(4.0, 40)])
    assert lat.made and lat.loaded and not lat.lost
    t = run(lat, t, [(0.0, 24)])              # a REAL collapse still ends the assist
    assert lat.lost and not lat.loaded
    print("PASS: contact latch rides out sub-grace force dips, breaks after grace")

    # --- recenter orientation: aim at the force, keep the measured roll ---
    # The push commands rx in FORCE mode with zero torque, so the real wrist roll is
    # compliant and the Gram-Schmidt roll was never a tracked setpoint. The assist
    # commands every axis in position, so re-aiming must not turn that roll into one.
    roll = 0.7
    R_meas = el._axis_angle_rot(np.array([1.0, 0.0, 0.0]), roll)
    q_meas = el.rotmat_to_quat_wxyz(R_meas)
    force = np.array([1.0, 0.2, 0.35])
    q_cmd = el.task_quat_with_measured_roll(force, q_meas)
    R_cmd = el.quat_wxyz_to_rotmat(q_cmd)
    assert np.allclose(R_cmd[:, 0], force / np.linalg.norm(force), atol=1e-9)
    # swing-only: the relative rotation carries no twist about the new x axis
    twist = Rotation.from_matrix(R_meas.T @ R_cmd).as_rotvec()
    assert abs(float(np.dot(twist, R_meas[:, 0]))) < 1e-9
    # a degenerate force leaves the measured orientation alone
    assert np.allclose(el.task_quat_with_measured_roll(np.zeros(3), q_meas), q_meas)

    # elevation clamp: bounds the pitch, keeps azimuth
    q_up = el.task_frame_from_force(np.array([1.0, 1.0, 1.2]))[1]
    elev_up = el.quat_elevation(q_up)
    assert elev_up > np.deg2rad(30.0)
    q_cl = el.clamp_quat_elevation(q_up, np.deg2rad(-5.0), np.deg2rad(10.0))
    assert abs(el.quat_elevation(q_cl) - np.deg2rad(10.0)) < 1e-9
    x_up = el.quat_wxyz_to_rotmat(q_up)[:, 0]
    x_cl = el.quat_wxyz_to_rotmat(q_cl)[:, 0]
    assert abs(np.arctan2(x_up[1], x_up[0]) - np.arctan2(x_cl[1], x_cl[0])) < 1e-9
    # inside the band it is a no-op
    assert np.allclose(el.clamp_quat_elevation(q_up, np.deg2rad(-90.0),
                                               np.deg2rad(90.0)), q_up)

    # slew limit: one step is never larger than the budget, and shrinks to nothing
    step = np.deg2rad(2.0)
    q_slew = el.slew_quat(q_meas, q_cmd, step)
    ang = np.linalg.norm(Rotation.from_matrix(
        R_meas.T @ el.quat_wxyz_to_rotmat(q_slew)).as_rotvec())
    assert abs(ang - step) < 1e-9
    assert np.allclose(el.slew_quat(q_meas, q_cmd, np.pi), q_cmd)   # budget not binding
    assert np.allclose(el.slew_quat(None, q_cmd, step), q_cmd)      # first tick
    print("PASS: assist orientation aims at the force, keeps the compliant roll, "
          "clamps elevation and slews")

    # --- body clearance: the y/z bands decide what blocks at all ---
    half_len, half_w, half_h = 0.55, 0.25, 0.10
    box = dict(front_margin=0.12, side_margin=0.05, over_margin=0.05,
               under_margin=0.05)
    front = half_len + box["front_margin"]
    wall = np.array([[0.0, y, z] for y in (-0.1, 0.0, 0.1) for z in (-0.05, 0.05)])
    I = np.eye(3)
    clr, hits = el.body_clearance_to_cloud(I, np.array([1.5, 0.0, 0.0]), wall,
                                           half_len, half_w, half_h, **box)
    assert len(hits) == len(wall) and abs(clr - (1.5 - front)) < 1e-9
    # an overhang Spot can walk under blocks nothing, whatever its x
    clr, hits = el.body_clearance_to_cloud(I, np.array([0.2, 0.0, 0.9]), wall,
                                           half_len, half_w, half_h, **box)
    assert np.isinf(clr) and len(hits) == 0
    # so does a cloud that passes under the belly, or one off to the side
    assert np.isinf(el.body_clearance_to_cloud(I, np.array([0.2, 0.0, -0.9]), wall,
                                               half_len, half_w, half_h, **box)[0])
    assert np.isinf(el.body_clearance_to_cloud(I, np.array([1.0, 1.2, 0.0]), wall,
                                               half_len, half_w, half_h, **box)[0])
    # already overlapping reads negative, so a gate on <= 0 fires
    clr, _ = el.body_clearance_to_cloud(I, np.array([0.5, 0.0, 0.0]), wall,
                                        half_len, half_w, half_h, **box)
    assert clr < 0.0
    # only the NEAREST blocking point counts
    two = np.vstack([wall, wall + np.array([1.0, 0.0, 0.0])])
    clr2, _ = el.body_clearance_to_cloud(I, np.array([1.5, 0.0, 0.0]), two,
                                         half_len, half_w, half_h, **box)
    assert abs(clr2 - (1.5 - front)) < 1e-9
    print("PASS: body clearance is the nearest blocking point; overhangs, belly "
          "clearance and off-axis clouds do not block")

    # --- mesh cloud: link-frame metres, seeded, on the surface ---
    npz = ("/home/rwl-4090/gits/nonprehensile_object_manipulation/dataset/primitives/"
           "Paralelopiped/Paralelopiped.npz")
    obj = os.path.splitext(npz)[0] + ".obj"
    if os.path.isfile(obj) and os.path.isfile(npz):
        cloud, n_verts, n_faces = el.load_mesh_cloud(obj, samples=500)
        assert cloud.shape == (n_verts + 500, 3) and n_faces > 0
        with np.load(npz) as z:
            link = (np.asarray(z["pts_norm"]) * float(z["scale"])
                    + np.asarray(z["centroid"]).reshape(3))
        # the mesh IS the link frame: pts_norm*scale + centroid undoes exactly the
        # normalisation the npz stores. The npz samples the surface, so its bbox sits
        # a hair inside the mesh bbox, never outside.
        assert np.abs(cloud.min(axis=0) - link.min(axis=0)).max() < 0.005
        assert np.abs(cloud.max(axis=0) - link.max(axis=0)).max() < 0.005
        # seeded: same file, same cloud, so an offline clearance number reproduces
        again, _, _ = el.load_mesh_cloud(obj, samples=500)
        assert np.array_equal(cloud, again)
        # every sample is a convex combination of one triangle's vertices, so no
        # sample may sit outside the vertex bounding box
        verts = cloud[:n_verts]
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        assert (cloud >= lo - 1e-9).all() and (cloud <= hi + 1e-9).all()
        # and the sampling is area-weighted, not vertex-clustered: the samples cover
        # the object rather than piling up wherever the mesh happens to be subdivided
        assert np.abs(cloud[n_verts:].mean(axis=0) - verts.mean(axis=0)).max() < 0.1
        print("PASS: mesh cloud is link-frame metres, seeded, area-weighted, "
              "and bounded by the mesh")
    else:
        print("SKIP: %s not present, mesh cloud test not run" % obj)

    # --- body-locked arm command: both stages of the assist ---
    tip, quat = [0.9, 0.0, 0.2], (1.0, 0.0, 0.0, 0.0)
    crawl = el.build_body_locked_arm_command(tip, quat)
    req = crawl.synchronized_command.arm_command.arm_cartesian_command
    assert req.root_frame_name == "flat_body"
    assert req.force_remain_near_current_joint_configuration
    folding = el.build_body_locked_arm_command(
        tip, quat, root_frame="odom", remain_near_current_joints=False,
        mobility_command=el.build_velocity_mobility_command(0.1))
    req = folding.synchronized_command.arm_command.arm_cartesian_command
    # An inertial root keeps the target put so the arm FOLDS as the body advances,
    # with the joint-configuration preference off so it is free to re-solve while it
    # does. RECENTER wants that fold but does NOT command it this way: an odom-rooted
    # arm command in the same SynchronizedCommand plants the body (see
    # build_body_locked_arm_command), so it maps its fixed odom point into flat_body
    # each tick instead. The root_frame argument still has to work both ways.
    assert req.root_frame_name == "odom"
    assert not req.force_remain_near_current_joint_configuration
    assert folding.synchronized_command.mobility_command.HasField(
        "se2_velocity_request")
    # ...which is what recenter actually sends: flat_body root, joint preference off,
    # base velocity riding along.
    recenter = el.build_body_locked_arm_command(
        tip, quat, remain_near_current_joints=False,
        mobility_command=el.build_velocity_mobility_command(0.1))
    req = recenter.synchronized_command.arm_command.arm_cartesian_command
    assert req.root_frame_name == "flat_body"
    assert not req.force_remain_near_current_joint_configuration
    assert recenter.synchronized_command.mobility_command.HasField(
        "se2_velocity_request")
    print("PASS: body-locked arm command roots in flat_body (crawl, recenter) or an "
          "inertial frame, with the joint-configuration preference following it")

    # --- base aim: heading that puts the crawl's line of action through the pivot ---
    # A synthetic slab, 1.0 x 0.6 x 0.4, standing on z=0 with its footprint centred
    # on the origin, so the pivot is known exactly.
    g = np.linspace(-0.5, 0.5, 21)
    h = np.linspace(-0.3, 0.3, 13)
    gx, gy = np.meshgrid(g, h)
    floor = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    slab = np.vstack([floor, floor + [0.0, 0.0, 0.4]])
    piv = el.support_pivot_xy(slab, 0.015)
    assert np.abs(piv).max() < 1e-9, piv
    # the top face must not enter the pivot: it has left the floor
    assert el.support_pivot_xy(slab[:floor.shape[0]], 0.015) is not None
    # tilt the slab about its leading edge (+x) by 30deg: the support collapses to
    # that edge, and the pivot moves onto it. This is the topple case.
    c, sn = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
    R = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
    tilted = (slab - [0.5, 0.0, 0.0]) @ R.T + [0.5, 0.0, 0.0]
    assert abs(el.support_pivot_xy(tilted, 0.015)[0] - 0.5) < 1e-6

    # centred contact (case b): the push axis ALREADY points at the pivot, so the
    # pivot rule must not move the heading. That case works on the robot today.
    aim, ang, clamped, why = el.aim_dir_to_pivot([-0.5, 0.0], piv, [1.0, 0.0, 0.0],
                                                 np.deg2rad(35.0))
    assert why == "" and not clamped and abs(ang) < 1e-9
    assert np.allclose(aim, [1.0, 0.0])
    # off-centre contact (cases a/c): the aim turns toward the pivot, and the moment
    # of a force along the aim about the pivot is zero - which is the whole point
    for y, sign in ((0.2, -1.0), (-0.2, 1.0)):
        contact = np.array([-0.5, y])
        aim, ang, clamped, why = el.aim_dir_to_pivot(contact, piv, [1.0, 0.0, 0.0],
                                                     np.deg2rad(35.0))
        assert why == "" and not clamped
        assert np.sign(ang) == sign and abs(np.rad2deg(ang) - sign * 21.8) < 0.1
        r = contact - piv
        assert abs(r[0] * aim[1] - r[1] * aim[0]) < 1e-12
        # ... whereas the push axis it replaces carries a real moment
        assert abs(r[0] * 0.0 - r[1] * 1.0) > 0.1
    # cone clamp: bounded, same side, and no longer through the pivot
    aim, ang, clamped, why = el.aim_dir_to_pivot([-0.2, 0.6], piv, [1.0, 0.0, 0.0],
                                                 np.deg2rad(35.0))
    assert clamped and why == "" and abs(np.rad2deg(ang) + 35.0) < 1e-9
    # degenerate inputs all fall back to the push axis, and say why
    for contact, frag in (([0.0, 0.0], "over pivot"), ([0.9, 0.0], "behind contact")):
        aim, ang, clamped, why = el.aim_dir_to_pivot(contact, piv, [1.0, 0.0, 0.0],
                                                     np.deg2rad(35.0))
        assert frag in why and np.allclose(aim, [1.0, 0.0]) and ang == 0.0
    aim, _, _, why = el.aim_dir_to_pivot([-0.5, 0.2], None, [1.0, 0.0, 0.0],
                                         np.deg2rad(35.0))
    assert why == "no pivot" and np.allclose(aim, [1.0, 0.0])
    assert el.aim_dir_to_pivot([-0.5, 0.2], piv, [0.0, 0.0, 1.0],
                               np.deg2rad(35.0))[0] is None
    print("PASS: pivot is the support-polygon centroid (and the tipping edge once "
          "tilted); the aim zeroes the crawl's yaw moment, clamps, and falls back")

    # --- yaw bookkeeping for the body-locked crawl ---
    assert abs(el.wrap_pi(np.deg2rad(370.0)) - np.deg2rad(10.0)) < 1e-9
    assert abs(el.wrap_pi(np.deg2rad(-350.0)) - np.deg2rad(10.0)) < 1e-9
    assert abs(el.signed_angle_xy([1.0, 0.0], [0.0, 1.0]) - np.pi / 2) < 1e-12
    # the tip is latched in flat_body, so a base yaw would sweep it across the face.
    # counter_yaw_tip cancels exactly that: the tip keeps its WORLD bearing from the
    # body origin, and its height and radius are untouched.
    tip0 = np.array([0.9, 0.0, 0.5])
    for dy in (np.deg2rad(12.0), -np.deg2rad(7.0), 0.0):
        tip_cmd = el.counter_yaw_tip(tip0, dy)
        assert abs(np.linalg.norm(tip_cmd[:2]) - np.linalg.norm(tip0[:2])) < 1e-12
        assert tip_cmd[2] == tip0[2]
        # rotate the command back into world by the base yaw: it lands on the latch
        world = el.rot_z_xy(tip_cmd[:2], dy)
        assert np.abs(world - tip0[:2]).max() < 1e-12
    assert np.array_equal(el.counter_yaw_tip(tip0, 0.0), tip0)
    print("PASS: counter_yaw_tip holds the latched tip's world bearing through a "
          "base yaw, and is inert at zero")



if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lib to build the hybrid-Cartesian ArmCartesianCommand for one push.
"""
from __future__ import annotations  # for 'Vec3 | None' python3.8

import numpy as np
from scipy.spatial.transform import Rotation
from bosdyn.client.frame_helpers import (ODOM_FRAME_NAME, VISION_FRAME_NAME,
                                         BODY_FRAME_NAME, HAND_FRAME_NAME,
                                         GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b)
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api import (arm_command_pb2, geometry_pb2, trajectory_pb2,
                        synchronized_command_pb2, robot_command_pb2,
                        basic_command_pb2, mobility_command_pb2, geometry_pb2)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.util import seconds_to_duration
from google.protobuf import wrappers_pb2

# Independent safety ceiling, not the policy's bound: the cone decode already
# caps commands at force_max = 75 N, so this only ever fires on a malformed
# goal. Fallback only, the executor node overrides it from ~exec/force_clip.
FORCE_NORM_CLIP = 75.0


def rotmat_to_quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def task_frame_from_force(push_force):
    """
    Return (magnitude, quat_wxyz) for a task frame whose x-axis = push dir.
    """
    f = np.asarray(push_force, dtype=np.float64)
    mag = float(np.linalg.norm(f))
    if mag < 1e-9:
        return 0.0, np.array([1.0, 0.0, 0.0, 0.0])
    x = f / mag
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    y = np.cross(up, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    return mag, rotmat_to_quat_wxyz(R)


def build_se2_mobility_command(x, y, yaw, root_frame=ODOM_FRAME_NAME,
                               max_lin_vel=0.5, max_ang_vel=0.75):
    """
    Mobility sub-command driving the base to an absolute SE2 pose (x, y, yaw) in
    root_frame, vel-limited. Returned as a MobilityCommand.Request so it can be
    embedded in a SynchronizedCommand alongside an arm command (an explicit base
    goal in place of FollowArmCommand). Built via RobotCommandBuilder so the
    MobilityParams (Any) packing is handled, then unwrapped to the sub-command.
    """
    speed = geometry_pb2.SE2VelocityLimit(
        max_vel=geometry_pb2.SE2Velocity(
            linear=geometry_pb2.Vec2(x=max_lin_vel, y=max_lin_vel),
            angular=max_ang_vel))
    params = spot_command_pb2.MobilityParams(vel_limit=speed)
    full = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        x, y, yaw, frame_name=root_frame, params=params)
    return full.synchronized_command.mobility_command


def build_stand_mobility_command():
    """
    Mobility sub-command that plants the base in place, as a MobilityCommand.Request
    for embedding in a SynchronizedCommand. Use it to positively cancel a running
    FollowArmCommand or SE2 goal: an arm-only command does NOT cancel a running
    mobility sub-command, so without this the base keeps executing the last thing it
    was told while the arm moves.
    """
    full = RobotCommandBuilder.synchro_stand_command()
    return full.synchronized_command.mobility_command


def build_se2_base_command(x, y, yaw, root_frame=ODOM_FRAME_NAME,
                           max_lin_vel=0.5, max_ang_vel=0.75):
    """
    Full mobility-ONLY RobotCommand driving the base to an absolute SE2 pose
    (x, y, yaw) in root_frame, vel-limited. Send this on its own low-rate cadence
    (~2-3 Hz) BESIDE a 40 Hz arm-only command: reissuing any locomotion goal at
    40 Hz supersedes the base step plan every tick so it never walks. An arm-only
    command does not cancel this running mobility, and vice versa, so the two
    coexist as independent sub-commands.
    """
    speed = geometry_pb2.SE2VelocityLimit(
        max_vel=geometry_pb2.SE2Velocity(
            linear=geometry_pb2.Vec2(x=max_lin_vel, y=max_lin_vel),
            angular=max_ang_vel))
    params = spot_command_pb2.MobilityParams(vel_limit=speed)
    return RobotCommandBuilder.synchro_se2_trajectory_point_command(
        x, y, yaw, frame_name=root_frame, params=params)


def build_arm_cartesian_command(contact_point, push_force, root_frame=ODOM_FRAME_NAME,
                                duration_s=1.5, force_clip=FORCE_NORM_CLIP,
                                follow_arm=False, body_offset_from_hand=None,
                                task_quat=None, tool_offset=(0.0, 0.0),
                                mobility_command=None):
    """
    Force mode that pushes along the given direction (task frame x axis), with lateral + rotational axes in position mode.
    mobility_command (MobilityCommand.Request), when given, is sent as the base
    sub-command (e.g. an explicit SE2 goal from build_se2_mobility_command); it
    takes precedence over follow_arm. follow_arm=True slaves the base to the arm;
    body_offset_from_hand (Vec3, only used when follow_arm) shifts where the base
    sits relative to the hand, e.g. to make the base recede behind a force-held
    hand during a coordinated retreat.
    task_quat (wxyz) pins the hand orientation independent of push_force; when
    given, only the magnitude is taken from push_force. This decouples the held
    orientation from a wrench being ramped to zero (which would otherwise collapse
    to the identity task frame). tool_offset (x, z) places the controlled tool
    frame at the pusher tip relative to the wrist, so the tip (not the wrist)
    tracks contact_point.
    Return (RobotCommand proto, magnitude, task_quat_wxyz).
    """

    mag, q = task_frame_from_force(push_force)
    if task_quat is not None:
        q = np.asarray(task_quat, dtype=np.float64)
    mag = min(mag, force_clip)
    cp = [float(v) for v in contact_point]
    tool_off_x, tool_off_z = tool_offset
    Req = arm_command_pb2.ArmCartesianCommand.Request

    identity = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    # tool frame at the pusher tip (wrist -> tip), so the tip tracks the command.
    wrist_tform_tool = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=tool_off_x, y=0.0, z=tool_off_z),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    root_tform_task = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=cp[0], y=cp[1], z=cp[2]),
        rotation=geometry_pb2.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]))

    pose_traj = trajectory_pb2.SE3Trajectory(points=[trajectory_pb2.SE3TrajectoryPoint(
        pose=identity, time_since_reference=seconds_to_duration(duration_s))])
    wrench_traj = trajectory_pb2.WrenchTrajectory(points=[trajectory_pb2.WrenchTrajectoryPoint(
        wrench=geometry_pb2.Wrench(force=geometry_pb2.Vec3(x=mag, y=0.0, z=0.0),
                                   torque=geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)),
        time_since_reference=seconds_to_duration(duration_s))])

    req = Req(
        root_frame_name=root_frame,
        root_tform_task=root_tform_task,
        wrist_tform_tool=wrist_tform_tool,
        pose_trajectory_in_task=pose_traj,
        wrench_trajectory_in_task=wrench_traj,
        x_axis=Req.AXIS_MODE_FORCE,
        y_axis=Req.AXIS_MODE_POSITION,
        z_axis=Req.AXIS_MODE_POSITION,
        # Roll about the push axis (task-x) is in force mode with zero torque 
        # (to keep it compliant) so the arm never exessively moves the wrist to match 
        # the random roll generated by gram-schmidt.
        rx_axis=Req.AXIS_MODE_FORCE,
        ry_axis=Req.AXIS_MODE_POSITION,
        rz_axis=Req.AXIS_MODE_POSITION,
    )
    arm_cmd = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=req)
    if mobility_command is not None:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_cmd, mobility_command=mobility_command)
    elif follow_arm:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_cmd,
            mobility_command=build_follow_arm_command(body_offset_from_hand))
    else:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd)
    cmd = robot_command_pb2.RobotCommand(synchronized_command=sync)
    return cmd, mag, q


def build_standoff_pose_command(contact_point, push_dir, quat_wxyz, root_frame=ODOM_FRAME_NAME,
                                eps=0.10, duration_s=2.0, follow_arm=False,
                                body_offset_from_hand=None, tool_offset=(0.0, 0.0),
                                mobility_command=None, max_lin_vel=None, max_accel=None):
    """
    Position-mode move that aligns the hand on the push axis before contact.
    mobility_command (MobilityCommand.Request), when given, is sent as the base
    sub-command (explicit SE2 goal) and takes precedence over follow_arm.
    follow_arm=True allows the base tracks the arm; body_offset_from_hand (Vec3,
    only used when follow_arm) shifts the base relative to the hand. tool_offset
    (x, z) places the controlled tool frame at the pusher tip relative to the
    wrist, so the tip stands off `eps` in front of the contact (not the wrist).
    eps is measured ALONG push_dir, so a negative eps puts the target that far
    PAST the contact (see build_approach_pose_command).

    max_lin_vel (m/s) and max_accel (m/s^2), when given, cap how fast the arm
    runs the move. Left unset the arm uses its own (fast) default, which is fine
    for a free-space align but not for driving into a surface.

    Returns (RobotCommand proto, standoff_point).
    """
    push_dir = np.asarray(push_dir, dtype=np.float64)
    n = np.linalg.norm(push_dir)
    if n > 1e-9:
        push_dir = push_dir / n
    standoff = np.asarray(contact_point, dtype=np.float64) - eps * push_dir
    sp = [float(v) for v in standoff]
    q = quat_wxyz
    tool_off_x, tool_off_z = tool_offset
    Req = arm_command_pb2.ArmCartesianCommand.Request

    identity = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    # tool frame at the pusher tip (wrist -> tip), so the tip stands off in front.
    wrist_tform_tool = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=tool_off_x, y=0.0, z=tool_off_z),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    root_tform_task = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=sp[0], y=sp[1], z=sp[2]),
        rotation=geometry_pb2.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]))
    pose_traj = trajectory_pb2.SE3Trajectory(points=[trajectory_pb2.SE3TrajectoryPoint(
        pose=identity, time_since_reference=seconds_to_duration(duration_s))])

    req = Req(
        root_frame_name=root_frame,
        root_tform_task=root_tform_task,
        wrist_tform_tool=wrist_tform_tool,
        pose_trajectory_in_task=pose_traj,
        x_axis=Req.AXIS_MODE_POSITION,
        y_axis=Req.AXIS_MODE_POSITION,
        z_axis=Req.AXIS_MODE_POSITION,
        rx_axis=Req.AXIS_MODE_POSITION,
        ry_axis=Req.AXIS_MODE_POSITION,
        rz_axis=Req.AXIS_MODE_POSITION,
    )
    # Optional wrappers: an unset DoubleValue means "arm default", a set one with
    # value 0 would mean "never move", so only touch these when asked to.
    if max_lin_vel is not None:
        req.max_linear_velocity.CopyFrom(
            wrappers_pb2.DoubleValue(value=float(max_lin_vel)))
    if max_accel is not None:
        req.maximum_acceleration.CopyFrom(
            wrappers_pb2.DoubleValue(value=float(max_accel)))
    arm_cmd = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=req)
    if mobility_command is not None:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_cmd, mobility_command=mobility_command)
    elif follow_arm:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_cmd,
            mobility_command=build_follow_arm_command(body_offset_from_hand))
    else:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd)
    cmd = robot_command_pb2.RobotCommand(synchronized_command=sync)
    return cmd, standoff


def build_carry_pose_command(hand_x, hand_z, root_frame=GRAV_ALIGNED_BODY_FRAME_NAME,
                             duration_s=2.5, max_lin_vel=None, max_accel=None,
                             mobility_command=None):
    """
    Position-mode move of the hand to a body-frame carry (tuck) pose at
    (hand_x, 0, hand_z), identity orientation. Built here rather than with
    RobotCommandBuilder.arm_pose_command so the rate caps below can be set:
    the retract starts from the extended arm_ready pose, near enough to the
    elbow-straight singularity that an uncapped Cartesian move asks for a large
    joint velocity on the first ticks.

    max_lin_vel (m/s) and max_accel (m/s^2), when given, cap the move; the accel
    cap is the one that removes the velocity spike at t=0. Left unset the arm
    uses its own (fast) default.

    mobility_command (MobilityCommand.Request), when given, rides along as the
    base sub-command; pass build_stand_mobility_command() to positively halt a
    still-running mobility goal, which an arm-only command does not cancel.

    Returns the RobotCommand proto.
    """
    Req = arm_command_pb2.ArmCartesianCommand.Request

    identity = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    root_tform_task = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=float(hand_x), y=0.0, z=float(hand_z)),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    pose_traj = trajectory_pb2.SE3Trajectory(points=[trajectory_pb2.SE3TrajectoryPoint(
        pose=identity, time_since_reference=seconds_to_duration(duration_s))])

    req = Req(
        root_frame_name=root_frame,
        root_tform_task=root_tform_task,
        pose_trajectory_in_task=pose_traj,
        x_axis=Req.AXIS_MODE_POSITION,
        y_axis=Req.AXIS_MODE_POSITION,
        z_axis=Req.AXIS_MODE_POSITION,
        rx_axis=Req.AXIS_MODE_POSITION,
        ry_axis=Req.AXIS_MODE_POSITION,
        rz_axis=Req.AXIS_MODE_POSITION,
    )
    # Optional wrappers: an unset DoubleValue means "arm default", a set one with
    # value 0 would mean "never move", so only touch these when asked to.
    if max_lin_vel is not None:
        req.max_linear_velocity.CopyFrom(
            wrappers_pb2.DoubleValue(value=float(max_lin_vel)))
    if max_accel is not None:
        req.maximum_acceleration.CopyFrom(
            wrappers_pb2.DoubleValue(value=float(max_accel)))
    arm_cmd = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=req)
    if mobility_command is not None:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_cmd, mobility_command=mobility_command)
    else:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd)
    return robot_command_pb2.RobotCommand(synchronized_command=sync)


def build_approach_pose_command(contact_point, push_dir, quat_wxyz, root_frame=ODOM_FRAME_NAME,
                                overshoot=0.02, duration_s=0.6, tool_offset=(0.0, 0.0),
                                mobility_command=None, max_lin_vel=0.05, max_accel=0.25):
    """
    Speed-limited all-position drive of the pusher tip into the contact.

    Unlike the force-mode push this exerts no commanded force: the tip travels at
    a bounded speed and the surface stops it, so the arrival is not an impact.
    That is the whole point - a force-mode approach across a standoff has nothing
    to react against, accelerates the entire way and lands at F = m_eff * a
    impulse instead of the commanded few Newtons.

    The target sits `overshoot` PAST the contact along push_dir so the position
    controller is still commanding motion when the tip meets the surface; a
    target exactly on the surface would decelerate to zero just short of it and
    never build the force the contact latch is watching for.

    Returns (RobotCommand proto, target_point).
    """
    return build_standoff_pose_command(
        contact_point, push_dir, quat_wxyz, root_frame=root_frame,
        eps=-abs(float(overshoot)), duration_s=duration_s, tool_offset=tool_offset,
        mobility_command=mobility_command, max_lin_vel=max_lin_vel,
        max_accel=max_accel)


def clamp_target_to_body(snap, root_frame, target_root, min_body_x):
    """
    Keep a pusher-tip target from being pulled back into the robot's own body.

    A retract target derived from the live object pose has no lower bound: an object
    that follows the tip drags the target backwards without limit. Clamping the
    target's gravity-aligned-body x against min_body_x bounds it geometrically,
    whatever the object does.

    snap is a bosdyn FrameTreeSnapshot; target_root is the TIP target expressed in
    root_frame (the command builders place the tool frame at the tip, so this is the
    same measure the reach gauge uses).

    Returns (clamped_target_root as np.ndarray, binding).
    """
    body_T_root = get_a_tform_b(snap, GRAV_ALIGNED_BODY_FRAME_NAME, root_frame)
    p_body = np.array(body_T_root.transform_point(*[float(v) for v in target_root]))
    if p_body[0] >= min_body_x:
        return np.asarray(target_root, dtype=np.float64), False
    p_body[0] = min_body_x
    root_T_body = get_a_tform_b(snap, root_frame, GRAV_ALIGNED_BODY_FRAME_NAME)
    return np.array(root_T_body.transform_point(*p_body)), True


def base_align_pose(contact_point, push_dir, d_base):
    # SE2 goal for the base such that it sits d_base behind the 
    # contact on the push axis. Contact point and push_dir are 
    # in the world frame.
    p = np.asarray(contact_point, dtype=np.float64)
    d = np.asarray(push_dir, dtype=np.float64)
    dir_xy = d[:2]
    n = np.linalg.norm(dir_xy)
    if n < 1e-6:
        return None
    dir_xy = dir_xy / n
    base_xy = p[:2] - d_base * dir_xy
    yaw = float(np.arctan2(dir_xy[1], dir_xy[0]))
    return float(base_xy[0]), float(base_xy[1]), yaw


def world_contact_from_object(R_root_obj, t_root_obj, push_point, force_body):
    """
    Transform the contact point and force vector from the object frame to the world frame.
    Returns (contact_world, force_world) as 3-vectors.
    """
    R = np.asarray(R_root_obj, dtype=np.float64)
    t = np.asarray(t_root_obj, dtype=np.float64)
    contact_world = R @ np.asarray(push_point, dtype=np.float64) + t
    force_world = R @ np.asarray(force_body, dtype=np.float64)
    return contact_world, force_world


def as_xyz(p):
    # Accept either a geometry_msgs/Point-like object or any 3-sequence.
    if hasattr(p, "x"):
        return np.array([p.x, p.y, p.z], dtype=np.float64)
    return np.asarray(p, dtype=np.float64).reshape(3)


def resolve_push_point(loc_idx, push_point, link_pts=None):
    """
    Object/link-frame contact point for one push goal, plus the path that produced it.

    Mirrors the sim, where the action's first element IS the contact specification:
    `location = action[:,0].long().clamp(0, 127)` indexes model_pcl_scaled_link_frame
    (objectmanip_env_discrete.py:1477,1539). link_pts is that same cloud on the ROS
    side, i.e. a cached obs_lib.link_frame_cloud() of shape (N, 3).

    loc_idx < 0 is the sentinel for the bench path: use the explicit push_point
    instead. Test rigs push the object ORIGIN, which is not a cloud point, so the
    index cannot express it.

    Returns (point (3,), source) with source in {"loc_idx", "push_point"}.
    Raises ValueError when loc_idx >= 0 but no cloud is loaded or the index is
    out of range; a silently substituted point would push the wrong face.
    """
    idx = int(loc_idx)
    if idx < 0:
        return as_xyz(push_point), "push_point"
    if link_pts is None:
        raise ValueError(
            "loc_idx=%d needs the object model cloud, but none is loaded "
            "(set ~npz_path)" % idx)
    pts = np.asarray(link_pts, dtype=np.float64)
    if idx >= pts.shape[0]:
        raise ValueError("loc_idx=%d out of range for a %d-point cloud"
                         % (idx, pts.shape[0]))
    return pts[idx].copy(), "loc_idx"


# Spot arm joint order as reported in robot_state.kinematic_state.joint_states.
_ARM_JOINT_NAMES = ("arm0.sh0", "arm0.sh1", "arm0.el0",
                    "arm0.el1", "arm0.wr0", "arm0.wr1")

# Product-of-exponentials kinematics, extracted directly from
# spot_description/urdf/spot_arm_macro.urdf (verified against the URDF joint
# origins + axes). Every arm joint origin in that URDF is a pure translation
# (rpy=0), and the fixed arm_hr0 joint has a zero offset, so the chain reduces
# to (parent-frame translation, local rotation axis) per joint. Row order
# matches _ARM_JOINT_NAMES. The final row is the fixed wr1->hand (arm_f1x) tool
# offset with no axis. Base offset body->sh0 is constant and cancels in
# det(J J^T), but is kept so FK origins are in the `body` frame.
_SPOT_ARM_POE = (
    #  offset in parent frame       local axis   name
    (np.array([0.292, 0.0, 0.188]), np.array([0.0, 0.0, 1.0])),  # sh0
    (np.array([0.0,   0.0, 0.0]),   np.array([0.0, 1.0, 0.0])),  # sh1
    (np.array([0.3385, 0.0, 0.0]),  np.array([0.0, 1.0, 0.0])),  # el0 (+fixed hr0)
    (np.array([0.4033, 0.0, 0.075]), np.array([1.0, 0.0, 0.0])), # el1
    (np.array([0.0,   0.0, 0.0]),   np.array([0.0, 1.0, 0.0])),  # wr0
    (np.array([0.0,   0.0, 0.0]),   np.array([1.0, 0.0, 0.0])),  # wr1
)
# Fixed wr1 -> hand tool tip (arm_f1x origin), no joint.
_SPOT_ARM_TOOL_OFFSET = np.array([0.11745, 0.0, 0.014820])

# Shoulder pivot (sh0/sh1 rotation center) in the `body` frame. sh1's own offset
# in _SPOT_ARM_POE is zero, so sh0 and sh1 share this origin and it IS the pivot
# the arm's reach spec is measured from.
_SPOT_SHOULDER_FROM_BODY = _SPOT_ARM_POE[0][0]


def shoulder_reach(tip_body):
    """
    Radial distance (m) from the shoulder pivot to a point given in the `body`
    frame; how far the arm is actually extended.

    This is the measure Boston Dynamics' reach number refers to (shoulder to
    gripper, radial), NOT the point's body-frame x. Body x is offset by the
    shoulder's own 0.292 m and throws away the y/z extension, so it reads a
    low or lateral push as far shorter than it is.

    Geometric maximum for the chain in _SPOT_ARM_POE:
        0.3385 (sh1->el0) + hypot(0.4033, 0.075) = 0.41022 (el0->wr1)
        + the wr1->tip offset (0.24187 for the 0.24/0.03 pusher tip)
        = 0.9906 m
    matching the published ~0.985 m arm reach.
    """
    return float(np.linalg.norm(np.asarray(tip_body, dtype=np.float64)
                                - _SPOT_SHOULDER_FROM_BODY))


def tip_reach_from_shoulder(snap, tip_from_hand):
    """
    Arm extension (m) of the pusher tip: shoulder_reach() of the tip, where the
    tip is tip_from_hand expressed in the `hand` frame.

    Rooted in `body`, not flat_body: _SPOT_SHOULDER_FROM_BODY is a URDF constant
    in `body`, so measuring against the gravity-aligned frame would add the
    body's own pitch/roll as reach error.

    snap is a bosdyn FrameTreeSnapshot.
    """
    body_T_hand = get_a_tform_b(snap, BODY_FRAME_NAME, HAND_FRAME_NAME)
    tip_body = body_T_hand.transform_point(*[float(v) for v in tip_from_hand])
    return shoulder_reach(tip_body)


def _axis_angle_rot(axis, theta):
    """3x3 rotation about a unit `axis` by `theta` (Rodrigues)."""
    ct, st = np.cos(theta), np.sin(theta)
    x, y, z = axis
    K = np.array([[0.0, -z,  y],
                  [z,  0.0, -x],
                  [-y,  x,  0.0]], dtype=np.float64)
    return np.eye(3) + st * K + (1.0 - ct) * (K @ K)


def arm_manipulability(joint_states):
    """
    Yoshikawa manipulability index w = sqrt(det(J J^T)) for the Spot arm, from
    the reported joint angles. Small w -> near a singularity (e.g. full
    extension). Returns nan if the arm joint angles are not all present.

    Kinematics come from _SPOT_ARM_POE, extracted from spot_description's URDF,
    so the geometric Jacobian is exact for bosdyn-reported joint angles.

    joint_states: the repeated JointState from
    robot_state.kinematic_state.joint_states (each has .name and .position.value).
    """
    angles = {js.name: js.position.value for js in joint_states}
    try:
        q = [float(angles[n]) for n in _ARM_JOINT_NAMES]
    except KeyError:
        return float("nan")

    # Forward kinematics in the body frame: for each joint apply the parent-frame
    # translation to reach the joint's link frame, record its world-frame axis +
    # origin (for the geometric Jacobian), then apply the joint rotation.
    R = np.eye(3)
    p = np.zeros(3)
    origins = np.zeros((6, 3))
    axes = np.zeros((6, 3))
    for i, (offset, axis_local) in enumerate(_SPOT_ARM_POE):
        p = p + R @ offset
        axes[i] = R @ axis_local            # rotation axis in body frame
        origins[i] = p                       # a point on that axis, body frame
        R = R @ _axis_angle_rot(axis_local, q[i])
    pe = p + R @ _SPOT_ARM_TOOL_OFFSET       # tool tip, body frame

    J = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        J[:3, i] = np.cross(axes[i], pe - origins[i])
        J[3:, i] = axes[i]
    w2 = float(np.linalg.det(J @ J.T))
    return float(np.sqrt(max(w2, 0.0)))


def build_follow_arm_command(body_offset_from_hand: geometry_pb2.Vec3 | None = None):
    if body_offset_from_hand is None:
        mobility_cmd = mobility_command_pb2.MobilityCommand.Request(
                        follow_arm_request=basic_command_pb2.FollowArmCommand.Request())
    else:
        mobility_cmd = mobility_command_pb2.MobilityCommand.Request(
                        follow_arm_request=basic_command_pb2.FollowArmCommand.Request(
                            body_offset_from_hand=body_offset_from_hand))
    return mobility_cmd

def tilt_since(R_ref, R_now):
    """
    Tilt (rad, in [0, pi]) the object has picked up since R_ref: the rotation from
    R_ref to R_now with its yaw (world +z) component removed. Both arguments are
    object rotations in the (gravity-aligned) root frame.

    Drives the topple detector: tipping accumulates tilt, while a pure slide or a
    spin-in-place reads zero. Measured against the push-start pose instead of an
    absolute angle-from-upright, so the gauge never assumes which body axis the
    calibrated object frame calls +z, nor which face the object rests on. An
    absolute gauge gets both wrong: a frame whose +z points down reports the tilt
    with the sign flipped, and a frame whose +z lies horizontal reports noise.
    """
    R_delta = (np.asarray(R_now, dtype=np.float64)
               @ np.asarray(R_ref, dtype=np.float64).T)
    x, y, z, w = Rotation.from_matrix(R_delta).as_quat()
    # Swing-twist split of the delta about world +z: writing delta = twist * swing,
    # the swing (tip) factor is (w*w + z*z, w*x - y*z, w*y + x*z, 0), all scaled by
    # the twist norm that atan2 divides back out. atan2 rather than the shorter
    # acos(hypot(w, z)) so a near-zero tilt does not land in the ill-conditioned
    # corner of acos; a pi tip lands on atan2(+, 0) and reads pi with no branch.
    twist_sq = w * w + z * z
    if twist_sq < 1e-18:
        # delta is a pi rotation about a horizontal axis: all tip, no yaw to split off
        return float(np.pi)
    return float(2.0 * np.arctan2(np.hypot(w * x - y * z, w * y + x * z), twist_sq))


def tilt_rise(hist, window):
    """
    (rise, valid) for a rolling [(t, tilt_rad), ...] history: how much tilt was gained
    over the last `window` seconds, and whether the history spans a full window yet.

    valid is False until it does, so a tip that has only just started cannot be judged
    on one sample; callers treat that as "no evidence of progress", same convention as
    the tip stall gauge.
    """
    if len(hist) < 2 or (hist[-1][0] - hist[0][0]) < window:
        return 0.0, False
    return float(hist[-1][1] - hist[0][1]), True


class ContactLatch(object):
    """
    Debounced contact state from the EE force estimate.
    made_n is the make threshold (arming), eps the break threshold; the band
    between them is hysteresis and reads as still in contact.
    """

    def __init__(self, made_n, eps, grace):
        self.made_n = float(made_n)
        self.eps = float(eps)
        self.grace = float(grace)
        self.made = False
        self.lost = False
        self.made_time = None   # wall time contact first armed
        self._low_since = None
        self._low_for = 0.0

    def update(self, force, now):
        """
        Fold one force sample (N, or None when the estimate is unavailable) taken
        at wall time `now` into the latch. Returns (made, lost).

        force=None carries no information about the contact, so it neither
        advances nor resets the low window.
        """
        if force is None:
            return self.made, self.lost
        force = float(force)
        if force > self.made_n and not self.made:
            self.made, self.made_time = True, now
        if force >= self.eps:
            self._low_since, self._low_for = None, 0.0
            return self.made, self.lost
        if not self.made:
            # Never in contact yet: approach-phase zeros are not a loss.
            return self.made, self.lost
        if self._low_since is None:
            self._low_since = now
        self._low_for = now - self._low_since
        if self._low_for > self.grace:
            self.lost = True
        return self.made, self.lost

    def force_made(self, now):
        """
        Arm the latch from a contact signal that is not the force estimate (the
        approach's blocked-tip gauge), so a real contact whose load never crosses
        made_n still starts the push instead of running out the approach clock.

        Idempotent: a latch already armed keeps its original made_time.
        """
        if not self.made:
            self.made, self.made_time = True, now
            self._low_since, self._low_for = None, 0.0
        return self.made

    @property
    def loaded(self):
        # In contact, counting the grace window: the tip is treated as still
        # loaded through a transient dip, so a one-tick dropout cannot block the
        # topple assist gate.
        return self.made and not self.lost

    @property
    def low_for(self):
        # Seconds the force has been continuously below eps; 0.0 when loaded.
        return self._low_for


def build_velocity_mobility_command(v_x, v_y=0.0, v_rot=0.0,
                                    max_lin_vel=0.3, max_ang_vel=0.3):
    """
    Mobility sub-command driving the base at a constant BODY-frame velocity, as a
    MobilityCommand.Request for embedding in a SynchronizedCommand.

    Unlike an SE2 trajectory goal, a velocity command carries no step plan for a
    reissue to supersede, so this one can safely be re-sent at loop_rate beside
    the arm command. The caller must still pass end_time_secs to robot_command();
    that is what fills se2_velocity_request.end_time (END_TIME_EDIT_TREE), and an
    unset end_time means the command lands already expired.
    """
    speed = geometry_pb2.SE2VelocityLimit(
        max_vel=geometry_pb2.SE2Velocity(
            linear=geometry_pb2.Vec2(x=max_lin_vel, y=max_lin_vel),
            angular=max_ang_vel))
    params = spot_command_pb2.MobilityParams(vel_limit=speed)
    full = RobotCommandBuilder.synchro_velocity_command(v_x, v_y, v_rot, params=params)
    return full.synchronized_command.mobility_command


def build_body_locked_arm_command(tip_point_body, quat_wxyz, duration_s=0.6,
                                  tool_offset=(0.0, 0.0), mobility_command=None,
                                  root_frame=GRAV_ALIGNED_BODY_FRAME_NAME,
                                  remain_near_current_joints=True):
    """
    All-position Cartesian command that pins the pusher tip to a FIXED pose in
    root_frame, turning the arm into a rigid strut. Pair it with a base velocity
    sub-command to push with the legs once the arm itself is out of travel (topple
    assist).

    root_frame picks which of the two assist stages this is:

    flat_body (default) - the target RIDES THE BODY, so the arm holds its shape and
        leg drive reaches the object through it. An odom-rooted target would stay
        put and the arm would fold as the body advanced. flat_body is
        gravity-aligned, so body pitch/roll does not drag the commanded tip height
        around.

    odom/vision - the target STAYS PUT while the body advances, so the arm folds in
        and recovers the travel it spent reaching. That is exactly the fold the
        flat_body mode avoids.

        DO NOT pair this root with a base velocity sub-command. Measured across bags
        spot_push_policy_8 and _10, an odom-rooted arm Cartesian command inside a
        SynchronizedCommand leaves the body PLANTED: forward travel along body +x was
        +0.001 m over 3.4 s against a commanded 0.10 m/s, with the feet never breaking
        contact and the arm joints moving under 3 deg, while the flat_body-rooted
        crawl immediately afterwards made +0.549 m. The recenter stage wants this
        fold, so it holds a point fixed in odom but commands it in flat_body, mapping
        the point through flat_T_root each tick - same physics, and the base walks.

    force_remain_near_current_joint_configuration (remain_near_current_joints)
    matters for the same reason. Locking happens at or near full extension, where
    the robot is free to pick a different preferred joint configuration and swing
    the arm through the singularity while it is loaded against the object - so the
    body-locked stage sets it. The recenter stage must NOT: it asks the arm to
    re-solve continuously as it folds in, which is the very thing the flag damps.

    tip_point_body is the tip position in root_frame, quat_wxyz the tool/wrist
    orientation in the same frame (wrist_tform_tool is a pure translation, so the
    tool and wrist share an orientation).

    Returns the RobotCommand proto.
    """
    p = [float(v) for v in tip_point_body]
    q = np.asarray(quat_wxyz, dtype=np.float64)
    tool_off_x, tool_off_z = tool_offset
    Req = arm_command_pb2.ArmCartesianCommand.Request

    identity = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    wrist_tform_tool = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=tool_off_x, y=0.0, z=tool_off_z),
        rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0))
    body_tform_task = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(x=p[0], y=p[1], z=p[2]),
        rotation=geometry_pb2.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]))
    pose_traj = trajectory_pb2.SE3Trajectory(points=[trajectory_pb2.SE3TrajectoryPoint(
        pose=identity, time_since_reference=seconds_to_duration(duration_s))])

    req = Req(
        root_frame_name=root_frame,
        root_tform_task=body_tform_task,
        wrist_tform_tool=wrist_tform_tool,
        pose_trajectory_in_task=pose_traj,
        force_remain_near_current_joint_configuration=bool(remain_near_current_joints),
        x_axis=Req.AXIS_MODE_POSITION,
        y_axis=Req.AXIS_MODE_POSITION,
        z_axis=Req.AXIS_MODE_POSITION,
        rx_axis=Req.AXIS_MODE_POSITION,
        ry_axis=Req.AXIS_MODE_POSITION,
        rz_axis=Req.AXIS_MODE_POSITION,
    )
    arm_cmd = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=req)
    if mobility_command is not None:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_cmd, mobility_command=mobility_command)
    else:
        sync = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd)
    return robot_command_pb2.RobotCommand(synchronized_command=sync)


def quat_wxyz_to_rotmat(quat_wxyz):
    # Inverse of rotmat_to_quat_wxyz. wxyz in, 3x3 out; scipy is xyzw.
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def swing_rotation(a, b):
    """
    Minimal (3x3) rotation taking unit vector a onto unit vector b: the rotation
    about their common perpendicular, with no twist about either. Applied to a
    whole frame it re-aims the frame's x axis while leaving the roll about that
    axis untouched, which is what keeps a re-aimed wrist from snapping.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        # Antiparallel: the common perpendicular is undefined, any axis normal to
        # a will do. Pick one that is not degenerate with a.
        axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        return _axis_angle_rot(axis / np.linalg.norm(axis), np.pi)
    return _axis_angle_rot(v / s, float(np.arctan2(s, c)))


def task_quat_with_measured_roll(force_world, measured_quat_wxyz):
    """
    Task quaternion whose x axis is the push force direction and whose ROLL about
    that axis is inherited from the measured wrist pose.

    task_frame_from_force fixes roll by Gram-Schmidt against world +z, which is
    fine while the push holds rx in force mode (the roll is compliant and the
    commanded value is never tracked). The assist commands every axis in position
    mode, so that Gram-Schmidt roll would suddenly become a real setpoint and the
    wrist would snap to it while loaded against the object. Re-aiming the measured
    frame instead changes only pitch and yaw.

    Returns the measured quaternion unchanged for a degenerate force.
    """
    f = np.asarray(force_world, dtype=np.float64).reshape(3)
    q_meas = np.asarray(measured_quat_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return q_meas
    R_meas = quat_wxyz_to_rotmat(q_meas)
    return rotmat_to_quat_wxyz(swing_rotation(R_meas[:, 0], f / n) @ R_meas)


def quat_elevation(quat_wxyz):
    # Elevation (rad) of the frame's x axis above the horizontal plane. The frame
    # must be gravity-aligned (odom/vision/flat_body) for this to mean anything.
    R = quat_wxyz_to_rotmat(quat_wxyz)
    return float(np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))


def clamp_quat_elevation(quat_wxyz, lo_rad, hi_rad):
    """
    Clamp the elevation of the frame's x axis into [lo, hi], keeping its azimuth
    and its roll about that axis.

    The commanded orientation is derived from a tracked object pose, so a dropped
    or mis-solved track can swing it a long way in one tick. This is the bound on
    how far it can go while the tool is loaded against the object.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    R = quat_wxyz_to_rotmat(q)
    x = R[:, 0]
    elev = float(np.arcsin(np.clip(x[2], -1.0, 1.0)))
    clamped = float(np.clip(elev, lo_rad, hi_rad))
    if abs(clamped - elev) < 1e-9:
        return q
    if np.hypot(x[0], x[1]) < 1e-9:
        # Straight up or down: azimuth is undefined, so there is no unique way to
        # tilt it back. Leave it, the caller's slew limit still bounds the motion.
        return q
    azim = float(np.arctan2(x[1], x[0]))
    x_clamped = np.array([np.cos(clamped) * np.cos(azim),
                          np.cos(clamped) * np.sin(azim),
                          np.sin(clamped)])
    return rotmat_to_quat_wxyz(swing_rotation(x, x_clamped) @ R)


def slew_quat(prev_quat_wxyz, target_quat_wxyz, max_step_rad):
    """
    target, limited to max_step_rad of rotation away from prev. Rate limit on the
    commanded orientation: the object pose it is derived from updates at mocap
    rate and can jump, the arm cannot.
    """
    target = np.asarray(target_quat_wxyz, dtype=np.float64).reshape(4)
    if prev_quat_wxyz is None or max_step_rad <= 0.0:
        return target
    R_prev = quat_wxyz_to_rotmat(prev_quat_wxyz)
    rotvec = Rotation.from_matrix(R_prev.T @ quat_wxyz_to_rotmat(target)).as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    if ang <= max_step_rad:
        return target
    step = Rotation.from_rotvec(rotvec * (max_step_rad / ang)).as_matrix()
    return rotmat_to_quat_wxyz(R_prev @ step)


def load_mesh_cloud(obj_path, samples=2000, seed=0):
    """
    Surface point cloud (M,3) from a Wavefront .obj, in the mesh's own coordinates.

    Deliberately a ~30-line parser rather than a trimesh/open3d dependency: the
    node's interpreter has neither, and this reads exactly the two record types a
    collision test needs. Vertex normals, textures, materials and groups are
    ignored; polygons are fan-triangulated; negative (relative) indices are
    resolved against the vertices seen so far, as the format specifies.

    The returned cloud is the vertices PLUS `samples` area-weighted points drawn
    over the faces. Vertices alone are exact for a convex support query, but a
    clearance test masks points by a y/z band and a large triangle can cross that
    band without putting a vertex in it. The sampling is seeded, so the same mesh
    always yields the same cloud and a clearance number is reproducible offline.

    Returns (cloud, n_vertices, n_faces).
    """
    verts, faces = [], []
    with open(obj_path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    head = tok.split("/")[0]
                    if not head:
                        continue
                    i = int(head)
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    V = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if V.shape[0] == 0:
        raise ValueError("no vertices in %s" % obj_path)
    if F.shape[0] == 0 or samples <= 0:
        return V, V.shape[0], F.shape[0]

    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total = float(area.sum())
    if total <= 0.0:
        return V, V.shape[0], F.shape[0]
    rng = np.random.RandomState(int(seed))
    pick = rng.choice(F.shape[0], size=int(samples), p=area / total)
    u = rng.random_sample((int(samples), 1))
    v = rng.random_sample((int(samples), 1))
    # Fold the unit square onto the unit triangle; uniform over each face.
    folded = (u + v) > 1.0
    u[folded] = 1.0 - u[folded]
    v[folded] = 1.0 - v[folded]
    P = V[F[pick, 0]] + u * e1[pick] + v * e2[pick]
    return np.vstack([V, P]), V.shape[0], F.shape[0]


def body_clearance_to_cloud(R_body_obj, t_body_obj, cloud_link,
                            half_len, half_width, half_height,
                            front_margin=0.0, side_margin=0.0,
                            over_margin=0.0, under_margin=0.0):
    """
    Forward clearance (m) between Spot's body box and an object point cloud, plus
    the points that produce it.

    R_body_obj/t_body_obj place the object's LINK frame in a gravity-aligned body
    frame (flat_body), so z is true height whatever the body is doing. The box is
    axis-aligned there: +-half_len along x, +-(half_width + side_margin) along y,
    and -(half_height + under_margin) .. +(half_height + over_margin) in z.

    A point only blocks the crawl if it lies inside the y and z bands. That is the
    whole trick: a cloud entirely ABOVE the box is an overhang Spot can walk under
    and does not limit travel at all, and one entirely below passes beneath the
    belly. Everything else reduces to how far forward the nearest blocking point
    is from the front face.

    Returns (clearance, blocking_points_in_body). clearance is +inf when nothing
    blocks, and negative once the box has already overlapped the cloud.
    """
    P = np.asarray(cloud_link, dtype=np.float64) @ np.asarray(
        R_body_obj, dtype=np.float64).T + np.asarray(
            t_body_obj, dtype=np.float64).reshape(3)
    blocking = ((np.abs(P[:, 1]) < (half_width + side_margin))
                & (P[:, 2] > -(half_height + under_margin))
                & (P[:, 2] < (half_height + over_margin)))
    if not np.any(blocking):
        return float("inf"), P[:0]
    hits = P[blocking]
    return float(hits[:, 0].min() - (half_len + front_margin)), hits


def wrap_pi(angle):
    # Fold an angle into (-pi, pi]. Heading errors are differences of two atan2
    # results, so the raw difference can be anywhere in (-2pi, 2pi).
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def signed_angle_xy(a, b):
    # Signed angle (rad) from 2-vector a to 2-vector b, +ve counterclockwise.
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a, b))))


def support_pivot_xy(cloud_root, support_eps=0.015):
    """
    xy of the object's SUPPORT POLYGON centroid, in whatever gravity-aligned frame
    the cloud is given in. This is the vertical axis the object yaws about.

    Not the link origin and not the cloud centroid. Yaw is resisted by friction over
    the patch the object actually stands on, so the pivot is the centroid of the
    points within support_eps of the lowest one. That single rule covers both cases
    the assist sees: a flat object gives the whole footprint, and a tipping one gives
    the leading edge it is rotating over, because everything else has left the floor.

    The link origin is NOT a usable substitute - Paralelopiped's sits 0.2m off its
    own footprint centre.

    Returns None for an empty cloud.
    """
    P = np.asarray(cloud_root, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] == 0:
        return None
    support = P[P[:, 2] <= (P[:, 2].min() + float(support_eps))]
    if support.shape[0] == 0:
        return None
    return support[:, :2].mean(axis=0)


def aim_dir_to_pivot(contact_xy, pivot_xy, push_dir, cone_rad, min_lever=0.05):
    """
    Base heading for an assist that must not yaw the object: the horizontal
    direction from the contact point to the object's yaw pivot.

    During the assist the arm is a rigid strut, so the force the legs deliver acts
    at the contact along body +x. Its moment about the object's vertical axis is
    ((p_contact - p_pivot) x F)_z, which is zero exactly when that line of action
    passes through the pivot. Facing the pivot therefore drives the object forward
    without spinning it, whatever the policy's own force direction was.

    Aiming down the push force instead - what base_align_pose alone does - re-applies
    the very moment that spun the object during the push.

    cone_rad bounds how far the aim may depart from push_dir: past that the crawl is
    driving the object somewhere the policy did not choose, and the standoff pose
    (still built along push_dir) walks off the body's x axis.

    Returns (aim_xy, ang_off, clamped, reason). aim_xy is a unit 2-vector and is
    always usable: on any degenerate input it falls back to the push direction and
    names the case in reason ("" when the pivot aim was used).
    """
    d = np.asarray(push_dir, dtype=np.float64).reshape(-1)[:2]
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return None, 0.0, False, "push near-vertical"
    d = d / n
    if pivot_xy is None:
        return d, 0.0, False, "no pivot"
    r = np.asarray(pivot_xy, dtype=np.float64).reshape(2) - np.asarray(
        contact_xy, dtype=np.float64).reshape(-1)[:2]
    lever = float(np.linalg.norm(r))
    if lever < float(min_lever):
        # Contact sits over the pivot: the direction is numerically meaningless and
        # the moment it would correct is already ~zero.
        return d, 0.0, False, "contact over pivot (%.3fm)" % lever
    aim = r / lever
    ang = signed_angle_xy(d, aim)
    if float(np.dot(aim, d)) <= 0.0:
        # The pivot is BEHIND the contact along the push: the goal contact is on the
        # far face, which no base heading fixes. Keep the push direction.
        return d, 0.0, False, "pivot behind contact (%.0fdeg)" % np.rad2deg(ang)
    if abs(ang) > float(cone_rad):
        ang = float(np.sign(ang) * cone_rad)
        c, s = np.cos(ang), np.sin(ang)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]), ang, True, ""
    return aim, ang, False, ""


def rot_z_xy(vec_xy, angle):
    # Rotate a 2-vector about +z by angle (rad).
    v = np.asarray(vec_xy, dtype=np.float64).reshape(2)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def counter_yaw_tip(tip_body_0, d_yaw):
    """
    The body-frame tip position that keeps a body-latched tip pointing in a FIXED
    WORLD direction while the base yaws by d_yaw.

    The crawl latches the tip rigidly in flat_body so leg drive reaches the object
    through the arm. That rigidity is what makes base yaw dangerous: a tip 0.9m
    ahead of the body sweeps 0.9*d_yaw sideways across the pushed face, which is a
    sliding contact, not a push. Rotating the latched offset back by the same yaw
    cancels exactly that: the tip keeps its world bearing from the body origin and
    still advances with every metre the body TRANSLATES, which is the part of the
    crawl that does the pushing.

    d_yaw is (body yaw now - body yaw at the latch). d_yaw = 0 returns the latch
    unchanged, so this is inert whenever the aim tracker is idle.
    """
    tip = np.asarray(tip_body_0, dtype=np.float64).reshape(3)
    xy = rot_z_xy(tip[:2], -float(d_yaw))
    return np.array([xy[0], xy[1], tip[2]])

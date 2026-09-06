#!/usr/bin/env python3
"""
Math and persistence for the <mocap body> -> object_link extrinsic. Pure
numpy/scipy, no ROS, so it is unit-testable offline and importable by the
calibration nodes.
"""
import os
import warnings

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from spot_calib.calib_lib import (make_se3, se3_inv, quat_wxyz_to_rotmat,
                                  rotmat_to_quat_wxyz)

# scipy lowercase == extrinsic, i.e. fixed-axis RPY about the parent (mocap
# body) axes. Same convention as `static_transform_publisher yaw pitch roll`,
# just in the opposite argument order.
RPY_CONVENTION = "xyz"

# Anything below this counts as "resting flat" for the floor residual to mean
# something; a tilted object makes min_z a meaningless target.
FLAT_TILT_LIMIT_DEG = 15.0

# Every rotation is within this angle of some cube-group element (the group's
# covering radius, 62.65 deg, measured not assumed), so a residual can never
# exceed it and the tuner's sliders can be ranged to cover it exactly once.
CUBE_COVERING_RADIUS_DEG = 62.65
# Re-seat the coarse element only once the residual is unambiguously past a
# Voronoi boundary. Re-seating on every drag would trade the pitch=+-90
# discontinuity for a boundary-crossing one; hysteresis removes both.
RESEAT_ANGLE_DEG = 30.0

# Fallback only. Every launch path passes ~parent_frame explicitly from the
# `object` arg; leaving the old body here makes an omission fail visibly rather
# than silently adopt whichever object is currently configured.
DEFAULT_PARENT = "object_1"
# NOT object_obs: that name belongs to npm_policy's identity-rotation render
# frame (obs_viz.OBS_FRAME). Sharing it gave the frame two parents and made the
# extrinsic look non-rigid in rviz.
DEFAULT_CHILD = "object_link"
YAML_KEY = "object_extrinsic"

__all__ = ["RPY_CONVENTION", "FLAT_TILT_LIMIT_DEG", "CUBE_COVERING_RADIUS_DEG",
           "RESEAT_ANGLE_DEG", "DEFAULT_PARENT", "DEFAULT_CHILD", "YAML_KEY",
           "make_se3", "se3_inv", "quat_wxyz_to_rotmat", "rotmat_to_quat_wxyz",
           "rpy_to_quat_wxyz", "quat_wxyz_to_rpy", "se3_from_xyz_rpy",
           "xyz_rpy_from_se3", "nearest_cube_rotation", "snap_rotation_to_90",
           "residual_rpy_from_se3", "se3_from_coarse_and_residual",
           "residual_angle_deg", "format_cube_rotation", "transform_points",
           "floor_residual", "load_calib", "save_calib", "frame_mismatch",
           "format_pose"]

_AXIS_NAMES = ("x", "y", "z")


def _as_euler(R, degrees=True):
    """as_euler with scipy's gimbal-lock warning suppressed.

    A calibrated mocap pivot usually lands on an exact cube-group element, and
    several of those sit on pitch=+-90. The absolute RPY is then genuinely
    ambiguous - which is why the tuner drives the residual instead - but it is
    still printed for reference, at the TF rate, so the warning would drown the
    log without telling the operator anything the readout does not.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Rotation.from_matrix(R).as_euler(RPY_CONVENTION, degrees=degrees)


def rpy_to_quat_wxyz(roll, pitch, yaw, degrees=True):
    return rotmat_to_quat_wxyz(
        Rotation.from_euler(RPY_CONVENTION, [roll, pitch, yaw],
                            degrees=degrees).as_matrix())


def quat_wxyz_to_rpy(q, degrees=True):
    return _as_euler(quat_wxyz_to_rotmat(q), degrees=degrees)


def se3_from_xyz_rpy(xyz, rpy, degrees=True):
    R = Rotation.from_euler(RPY_CONVENTION, np.asarray(rpy, dtype=np.float64),
                            degrees=degrees).as_matrix()
    return make_se3(R, np.asarray(xyz, dtype=np.float64).reshape(3))


def xyz_rpy_from_se3(T, degrees=True):
    """Absolute fixed-axis RPY. Display only - see residual_rpy_from_se3.

    This decomposition is singular at pitch=+-90, where roll and yaw collapse
    into one DOF, and the calibrated pose of an axis-aligned mocap pivot lands
    exactly there. Nothing the operator edits may be routed through it.
    """
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), _as_euler(T[:3, :3], degrees=degrees)


def _cube_group():
    """The 24 rotations of the cube, as 3x3 matrices.

    Built by brute force over signed axis permutations and filtered to det=+1;
    enumerating is cheaper than hand-writing 24 matrices correctly.
    """
    mats = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in np.ndindex(2, 2, 2):
            R = np.zeros((3, 3))
            for row, col in enumerate(perm):
                R[row, col] = 1.0 - 2.0 * signs[row]
            if np.isclose(np.linalg.det(R), 1.0):
                mats.append(R)
    return mats


_CUBE_GROUP = _cube_group()


def nearest_cube_rotation(R):
    """The cube-group element closest to R, as a 3x3 matrix.

    Doubles as the coarse half of the extrinsic: the mocap pivot is mounted
    face-aligned to the model, so this is the operator's intended gross
    orientation and everything left over is the misalignment to tune.
    """
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    # Nearest in Frobenius norm; maximizing trace(R^T C) is the same ranking.
    return max(_CUBE_GROUP, key=lambda C: float(np.trace(R.T @ C))).copy()


def snap_rotation_to_90(T):
    """Round the rotation to the nearest 90-degree-multiple orientation.

    The Motive pivot is near-axis-aligned with the model link frame in practice,
    so the true rotation is almost always an exact element of the cube group.
    Snapping turns an eyeballed couple of degrees into zero. Translation is left
    alone; only the operator knows where the origin sits.
    """
    T = np.asarray(T, dtype=np.float64)
    return make_se3(nearest_cube_rotation(T[:3, :3]), T[:3, 3])


def residual_rpy_from_se3(T, R_coarse, degrees=True):
    """Split the extrinsic as R = R_coarse @ R_residual and return xyz, residual RPY.

    Absolute RPY is unusable as an edit handle here: an axis-aligned pivot sits
    exactly on pitch=+-90, where roll and yaw stop being independent, so one
    slider cannot reach the third DOF and an infinitesimal drag rewrites the
    other two by 180 deg. Factoring the cube-group element out first leaves a
    residual near identity, decades away from the singularity, and it is the
    small misalignment the operator actually wants to trim.
    """
    T = np.asarray(T, dtype=np.float64)
    R_res = np.asarray(R_coarse, dtype=np.float64)[:3, :3].T @ T[:3, :3]
    return T[:3, 3].copy(), _as_euler(R_res, degrees=degrees)


def se3_from_coarse_and_residual(R_coarse, xyz, rpy, degrees=True):
    """Inverse of residual_rpy_from_se3; R_coarse must be the same element."""
    R_res = Rotation.from_euler(RPY_CONVENTION,
                                np.asarray(rpy, dtype=np.float64),
                                degrees=degrees).as_matrix()
    return make_se3(np.asarray(R_coarse, dtype=np.float64)[:3, :3] @ R_res,
                    np.asarray(xyz, dtype=np.float64).reshape(3))


def residual_angle_deg(T, R_coarse):
    """Geodesic angle of the residual rotation.

    The re-seat decision uses this rather than max(|rpy|) because Euler
    magnitude is not a metric: a rotation of a few degrees can decompose into
    three large angles, which would trigger spurious re-seats.
    """
    R_res = np.asarray(R_coarse, dtype=np.float64)[:3, :3].T @ \
        np.asarray(T, dtype=np.float64)[:3, :3]
    cos = (float(np.trace(R_res)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def format_cube_rotation(R):
    """Name a cube element by where it sends each child axis, e.g. "x->-z y->+x z->-y".

    Euler angles are exactly what this split stops showing the operator, so
    reporting the coarse element as RPY would reintroduce the ambiguity in the
    readout.
    """
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    out = []
    for col, name in enumerate(_AXIS_NAMES):
        v = R[:, col]
        row = int(np.argmax(np.abs(v)))
        out.append("%s->%s%s" % (name, "-" if v[row] < 0 else "+",
                                 _AXIS_NAMES[row]))
    return " ".join(out)


def transform_points(T, points):
    # Row-vector convention: (N,3) @ R.T + t, matching obs_lib.rotate_cloud.
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def floor_residual(link_pts, T_world_parent, T_parent_link, sunk_tol=0.005):
    """Objective slice of the alignment error, valid while the object rests flat.

    The `world` origin sits on the ground plane and the model's link origin sits
    on its bottom face, so a correctly calibrated flat-resting object puts the
    cloud's minimum world z at 0 with the link +z axis along world +z. That pins
    t_z and 2 of 3 rotation DOF; yaw and t_x/t_y stay the operator's problem.

    Returns min_z, tilt_deg, n_below (points sunk more than sunk_tol below the
    floor) and flat (whether tilt is small enough for min_z to mean anything).
    """
    T_world_link = np.asarray(T_world_parent, dtype=np.float64) @ np.asarray(
        T_parent_link, dtype=np.float64)
    world_pts = transform_points(T_world_link, link_pts)

    z_axis = T_world_link[:3, 2]
    cos_tilt = float(np.clip(z_axis[2] / max(np.linalg.norm(z_axis), 1e-12),
                             -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
    min_z = float(world_pts[:, 2].min())
    return {"min_z": min_z,
            "tilt_deg": tilt_deg,
            "n_below": int(np.count_nonzero(world_pts[:, 2] < -sunk_tol)),
            "flat": bool(tilt_deg <= FLAT_TILT_LIMIT_DEG)}


def load_calib(path, key=YAML_KEY):
    """Return (T 4x4, meta dict). A missing or uncalibrated file yields identity.

    Never raises on absence: the runtime publisher has to come up either way and
    say so loudly, rather than take the stack down over a first-run file.
    """
    if not path or not os.path.isfile(path):
        return np.eye(4), {"calibrated": False, "missing": True, "path": path}

    with open(path, "r") as fh:
        doc = yaml.safe_load(fh) or {}
    block = doc.get(key)
    if not block:
        return np.eye(4), {"calibrated": False, "missing": True, "path": path}

    t = block["translation"]
    t = [t["x"], t["y"], t["z"]] if isinstance(t, dict) else t
    T = make_se3(quat_wxyz_to_rotmat(block["rotation_wxyz"]), t)
    meta = {k: v for k, v in block.items()
            if k not in ("translation", "rotation_wxyz")}
    meta.setdefault("calibrated", False)
    meta["missing"] = False
    meta["path"] = path
    return T, meta


def frame_mismatch(meta, parent, child):
    """Frames the file was solved in vs the frames a node is about to publish.

    Returns a description of the disagreement, or None when they match. Both
    calib nodes take the frame names from params but the transform from the
    file, so pointing one object's launch args at another object's calibration
    silently republishes the wrong rotation under the right name.
    """
    if meta.get("missing"):
        return None
    bad = [(k, want, meta.get(k)) for k, want in (("parent_frame", parent),
                                                  ("child_frame", child))
           if meta.get(k) is not None and meta.get(k) != want]
    if not bad:
        return None
    return ", ".join("%s: file has %r, node wants %r" % (k, got, want)
                     for k, want, got in bad)


def save_calib(path, T, parent=DEFAULT_PARENT, child=DEFAULT_CHILD, **meta):
    """Write the extrinsic in spot_calib's block schema, creating parent dirs."""
    T = np.asarray(T, dtype=np.float64)
    block = {"parent_frame": parent,
             "child_frame": child,
             "translation": [float(v) for v in T[:3, 3]],
             "rotation_wxyz": [float(v) for v in rotmat_to_quat_wxyz(T[:3, :3])]}
    # Bookkeeping keys (calibrated, note, npz_path, floor_residual) round-trip
    # through load_calib's meta, so they are free-form on purpose.
    block.update({k: v for k, v in meta.items() if v is not None})

    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump({YAML_KEY: block}, fh, default_flow_style=False,
                       sort_keys=False)
    return block


def format_pose(T, R_coarse=None, degrees=True):
    """One-line pose, plus the coarse/residual split when R_coarse is given.

    The absolute RPY stays on the first line because that is what the saved
    quaternion and any external tool will report; the second line is what the
    tuner's sliders are actually driving.
    """
    xyz, rpy = xyz_rpy_from_se3(T, degrees=degrees)
    unit = "deg" if degrees else "rad"
    text = ("xyz [%.4f %.4f %.4f] m  rpy [%.2f %.2f %.2f] %s"
            % (xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2], unit))
    if R_coarse is not None:
        _, res = residual_rpy_from_se3(T, R_coarse, degrees=degrees)
        text += ("\ncoarse [%s]  residual rpy [%.2f %.2f %.2f] %s"
                 % (format_cube_rotation(R_coarse), res[0], res[1], res[2],
                    unit))
    return text

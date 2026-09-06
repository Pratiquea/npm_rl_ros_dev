#!/usr/bin/env python3
"""
Assemble the 798-dim SMDP observation. Pure numpy/scipy, no ROS and no torch,
so it is unit-testable offline and importable by the debug nodes.

Layout (raw/smdp_contract.md):
  obs (798) = [ pc(384) | normals(384) | extras(30) ]
"""
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

N_POINTS = 128
POINT_DIM = 3
PC_LEN = N_POINTS * POINT_DIM          # 384
EXTRAS_LEN = 30
OBS_LEN = 2 * PC_LEN + EXTRAS_LEN      # 798

# Block offsets in the full observation.
PC_OFF = 0
NORMALS_OFF = PC_LEN                   # 384
EXTRAS_OFF = 2 * PC_LEN                # 768

# Field offsets inside the 30-element extras block. Nothing may slice extras
# with a literal; the coordinator used to and it made the layout unrenamable.
# 4: [loc_idx, |F|/force_max, u1, u2] - the DECODED cone action, not the
# net's pre-squash output (objectmanip_env_discrete.py:1754-1755).
E_PREV_ACTION = 0
E_MASS = 4                             # 1
E_OBJECT_POSE = 5                      # 7: xyz (rel. goal) + quat wxyz
E_LINK_POSE = 12                       # 7
E_LIN_VEL = 19                         # 3
E_ANG_VEL = 22                         # 3
E_FRICTION = 25                        # 1
E_OBJECT_SCALE = 26                    # 1
E_DIST_TO_GOAL = 27                    # 1
E_DIR_TO_GOAL = 28                     # 2


def load_model_npz(path):
    """The one object-model loader for the workspace.

    Returns pts_norm (N,3), normals (N,3), scale (float), centroid (3,).
    pts_norm feeds the observation unscaled; the contact point the policy emits
    is pts_norm[loc]*scale + centroid, which is why centroid comes along.
    """
    with np.load(path) as z:
        pts = np.asarray(z["pts_norm"], dtype=np.float64)
        nrm = np.asarray(z["normals"], dtype=np.float64)
        scale = float(z["scale"])
        centroid = np.asarray(z["centroid"], dtype=np.float64).reshape(3)
    assert pts.shape == (N_POINTS, POINT_DIM), pts.shape
    assert nrm.shape == (N_POINTS, POINT_DIM), nrm.shape
    return pts, nrm, scale, centroid


def link_frame_cloud(pts_norm, scale, centroid, shrink=1.0):
    """Model points in the object's LINK frame: pts_norm*scale*shrink + centroid.

    The npz stores points normalized about the COM, but mocap tracks the link
    origin, so anything placed at the mocap pose must add the centroid (the COM
    origin expressed in link frame) back. shrink=1.0 mirrors the sim's
    model_pcl_scaled_link_frame; the observation deliberately does NOT use this.

    shrink < 1 is deploy-only margin: the scaling happens BEFORE the centroid is
    added, so the contraction centre is the COM itself. The COM does not move and
    the extrinsic needs no compensation; every point simply sinks below the real
    surface, which keeps the pointer argmax off the physical edges. The
    observation's object_scale slot keeps the nominal npz scale, because that is
    what the policy trained on.
    """
    return np.asarray(pts_norm, dtype=np.float64) * (float(scale) * float(shrink)) \
        + np.asarray(centroid, dtype=np.float64).reshape(3)


def shrink_inset(pts_norm, scale, shrink):
    # Per-axis inset (m) a shrink factor buys: half the nominal extent times
    # (1 - shrink). Logged at startup so the operator reads millimetres of margin
    # instead of a bare ratio.
    # This is the MEAN of the two sides, not each of them. The cloud contracts
    # about the COM, and the COM is not the centre of the bounding box (23 mm off
    # in y for Paralelopiped), so the two faces on one axis move in by slightly
    # different amounts: 12.7 and 11.3 mm where this reports 12.0. Good enough for
    # a log line; do not use it where a per-face margin has to be exact.
    pts = np.asarray(pts_norm, dtype=np.float64)
    half_extent = 0.5 * (pts.max(axis=0) - pts.min(axis=0)) * float(scale)
    return half_extent * (1.0 - float(shrink))


def rotate_cloud(rot, points):
    # Rotation.apply computes points @ R.T, same as sim's bmm(P, R.T).
    return np.asarray(rot.apply(np.asarray(points, dtype=np.float64)),
                      dtype=np.float64).reshape(-1).astype(np.float32)


def quat_wxyz(rot):
    # The obs and the policy use Isaac wxyz; scipy (and ROS) use xyzw.
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def goal_metrics(pos_world):
    # The goal is the world origin, so distance/direction are read off the pose.
    xy = np.asarray(pos_world, dtype=np.float64).reshape(3)[:2]
    dist = float(np.linalg.norm(xy))
    return dist, (-xy / max(dist, 1e-6))


def assemble_observation(rot, pos_world, lin_vel, ang_vel, prev_action, mass,
                         friction, object_scale, pts_norm, normals):
    """Return the float32 (798,) observation.

    rot: scipy Rotation, object orientation in `world`.
    pos_world: object position in `world`, which is also its position relative
    to the goal. The point cloud is rotated but deliberately NOT translated
    (object-centered, world-aligned), exactly as the sim builds it.
    """
    pc = rotate_cloud(rot, pts_norm)
    nrm = rotate_cloud(rot, normals)

    pos = np.asarray(pos_world, dtype=np.float64).reshape(3)
    dist, dir_to_goal = goal_metrics(pos)

    object_pose = np.concatenate([pos, quat_wxyz(rot)])
    # v1: mocap tracks a single rigid body, so the COM and link poses coincide.
    link_pose = object_pose.copy()

    extras = np.concatenate([
        np.asarray(prev_action, dtype=np.float64).reshape(4),
        np.array([mass], dtype=np.float64),
        object_pose,
        link_pose,
        np.asarray(lin_vel, dtype=np.float64).reshape(3),
        np.asarray(ang_vel, dtype=np.float64).reshape(3),
        np.array([friction], dtype=np.float64),
        np.array([object_scale], dtype=np.float64),
        np.array([dist], dtype=np.float64),
        dir_to_goal,
    ]).astype(np.float32)
    assert extras.shape[0] == EXTRAS_LEN, extras.shape

    obs = np.concatenate([pc, nrm, extras]).astype(np.float32)
    assert obs.shape[0] == OBS_LEN, obs.shape
    return obs


def rotation_from_ros_quat(q):
    # geometry_msgs/Quaternion -> scipy Rotation, the single xyzw boundary.
    return Rotation.from_quat([q.x, q.y, q.z, q.w])


def model_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def format_model_info(npz_path, scale, shrink=1.0):
    # Payload of the latched /npm/model_info topic the policy publishes. The
    # fourth field is the EFFECTIVE scale (scale*shrink), not the npz value: two
    # nodes holding the same file but different pcl_shrink resolve one loc_idx to
    # two different points, and the sha1 alone cannot see that.
    return "%s %s %d %.6f" % (npz_path, model_sha1(npz_path), N_POINTS,
                              float(scale) * float(shrink))


def npz_scale(path):
    # Just the scale, for callers that check /npm/model_info without paying for
    # the arrays.
    with np.load(path) as z:
        return float(z["scale"])


def check_model_info(data, npz_path, shrink=1.0, tol=1e-6):
    """(ok, message) for one /npm/model_info payload against a local .npz.

    loc_idx is only an index, so two nodes holding different models resolve it to
    different points and push the wrong face with nothing failing. ok=False is an
    error worth shouting about, not a warning.

    The effective scale is checked alongside the sha1: an identical file with a
    different pcl_shrink is exactly the same failure, only harder to see.
    """
    parts = str(data).split()
    if len(parts) < 2:
        return False, "unparsable /npm/model_info %r" % (data,)
    theirs, mine = parts[1], model_sha1(npz_path)
    if theirs != mine:
        return False, ("OBJECT MODEL MISMATCH: policy has %s (%s), this node has %s (%s). "
                       "loc_idx goals will push the WRONG face."
                       % (parts[0], theirs[:8], npz_path, mine[:8]))

    if len(parts) < 4:
        return True, ("object model agrees with the policy (sha1 %s); payload "
                      "carries no effective scale, pcl_shrink UNCHECKED" % mine[:8])
    try:
        theirs_eff = float(parts[3])
    except ValueError:
        return False, "unparsable effective scale in /npm/model_info %r" % (data,)
    mine_eff = npz_scale(npz_path) * float(shrink)
    if abs(theirs_eff - mine_eff) > tol:
        return False, ("PCL_SHRINK MISMATCH: policy resolves loc_idx at effective "
                       "scale %.6f, this node at %.6f (pcl_shrink=%.4f). Same file, "
                       "different contact points; fix pcl_shrink in npm.yaml."
                       % (theirs_eff, mine_eff, float(shrink)))
    return True, ("object model agrees with the policy (sha1 %s, effective scale "
                  "%.6f)" % (mine[:8], mine_eff))

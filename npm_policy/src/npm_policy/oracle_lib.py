#!/usr/bin/env python3
"""
Scripted oracle action, ported from objectmanip_env_discrete._oracle_action
(which now raises NotImplementedError upstream, so this is the only copy).
Pure numpy/scipy, no ROS and no torch.

Pick the surface point whose outward normal points most away from the goal,
prefer the tallest such point, and push it back toward the goal with a slight
upward tilt.
"""
import math

import numpy as np

from npm_policy import policy_lib as pl
from npm_policy.obs_lib import N_POINTS, link_frame_cloud

TOPK_NORMAL = 30
TOPK_HEIGHT = 10
FORCE_DIR_Z = 0.35            # upward tilt added to the goal-ward push direction
ORACLE_FORCE_MAG = pl.FORCE_MAX

# |u| must stay strictly inside the unit disk or the inverse squash divides by
# zero; this caps the encoded direction just short of the cone boundary.
U_MAX = 0.999
# The sigmoid never reaches its asymptotes, so a request for exactly FORCE_MAX
# comes back about 7e-5 N short. Tighten only if that ever matters.
SIGMOID_EPS = 1e-6

# Deliberately NOT ported, because they are commented out inside the sim's
# _oracle_action and would change the action:
#   _compute_reachable_mask   (sim:1136-1138)  ground/normal reachability filter
#   _sample_from_topk         (sim:1141)       random pick among the top-k scores
#   randint over the height candidates (sim:1168-1169)
# Also dropped, as sim-only domain randomisation applied outside the function:
#   noise.perturb_force (sim:1460), use_oracle_mixed (sim:1425-1431).


def _normalize(v, eps=1e-6):
    return np.asarray(v, dtype=np.float64) / max(float(np.linalg.norm(v)), eps)


def cone_encode(force_dir_body, force_mag, n_hat, up_body, goal_body, u_max=U_MAX):
    """Inverse of policy_lib.decode_cone_action: the pre-squash (a_m, a_tan) that
    decodes to the closest in-cone approximation of force_dir_body at force_mag.

    Exact when the desired direction lies inside the cone; otherwise it lands on
    the cone boundary. All vectors are in the object body frame.
    """
    t1, t2 = pl.tangent_basis_anchored(n_hat, up_body, goal_body)
    n_hat = np.asarray(n_hat, dtype=np.float64).reshape(3)
    d = _normalize(force_dir_body)

    # {-n_hat, t1, t2} is orthonormal, so scaling d to unit extent along -n_hat
    # reads the tangent-disk coordinates straight off the projections.
    c = float(np.dot(d, -n_hat))
    if c <= 1e-6:
        # Desired direction pulls away from the surface; push straight in.
        u = np.zeros(2, dtype=np.float64)
    else:
        p = np.array([float(np.dot(d, t1)) / c, float(np.dot(d, t2)) / c],
                     dtype=np.float64) / pl.CONE_TAN_THETA_MAX
        m = float(np.linalg.norm(p))
        u = p if m <= u_max else p * (u_max / m)

    a_tan = u / math.sqrt(max(1.0 - float(np.dot(u, u)), 1e-12))

    s = float(np.clip((force_mag - pl.FORCE_MIN) / (pl.FORCE_MAX - pl.FORCE_MIN),
                      SIGMOID_EPS, 1.0 - SIGMOID_EPS))
    a_m = math.log(s / (1.0 - s))
    return a_m, a_tan


def oracle_raw_action(rot, pos_world, pts_norm, normals, scale, centroid,
                      topk=TOPK_NORMAL, height_topk=TOPK_HEIGHT,
                      force_dir_z=FORCE_DIR_Z, force_mag=ORACLE_FORCE_MAG,
                      shrink=1.0):
    """Raw action [loc_idx, a_m, a1, a2], the same layout the policy net emits.

    rot: scipy Rotation, object orientation in `world`. pos_world doubles as the
    position relative to the goal, which is the world origin.

    The continuous slots are pre-squash cone coordinates, so policy_lib.postprocess
    decodes this exactly as it decodes a policy action. The desired push direction
    is clamped into the 20 deg cone about the inward normal, which makes oracle
    pushes markedly more normal-aligned than the old hemispherical oracle's.
    """
    R = np.asarray(rot.as_matrix(), dtype=np.float64)
    pos = np.asarray(pos_world, dtype=np.float64).reshape(3)
    normals = np.asarray(normals, dtype=np.float64)

    normals_world = normals @ R.T
    # Same shrunk cloud the executor will aim at, so the height tie-break ranks
    # the points that are actually commandable.
    pts_world = link_frame_cloud(pts_norm, scale, centroid, shrink) @ R.T + pos

    # Points on the far side of the object from the goal score highest, so the
    # push runs through the object rather than across it.
    dir_away = _normalize([pos[0], pos[1], 0.0])
    scores = normals_world @ dir_away

    k = min(int(topk), N_POINTS)
    kh = min(int(height_topk), k)
    # Stable descending sorts, matching torch.topk's low-index tie-break.
    by_score = np.argsort(-scores, kind="stable")[:k]
    by_height = np.argsort(-pts_world[by_score, 2], kind="stable")[:kh]
    loc = int(by_score[by_height[0]])

    # Toward the goal in xy, tilted up: a purely horizontal push at this height
    # topples the object instead of sliding it.
    force_dir = _normalize(-pos)
    force_dir[2] = force_dir_z
    force_dir = _normalize(force_dir)

    up_body, goal_body = pl.cone_frame_anchors(rot, pos)
    a_m, a_tan = cone_encode(R.T @ force_dir, force_mag, normals[loc],
                             up_body, goal_body)

    action = np.empty(4, dtype=np.float64)
    action[0] = float(loc)
    action[1] = a_m
    action[2:4] = a_tan
    return action

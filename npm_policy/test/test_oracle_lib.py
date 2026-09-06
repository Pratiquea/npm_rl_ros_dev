#!/usr/bin/env python3
"""
Offline invariant test for oracle_lib (no ROS, no torch, no checkpoint). Runs on
a synthetic box cloud so it does not depend on any dataset file.

Example:
    /usr/bin/python3 test_oracle_lib.py
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from npm_policy import oracle_lib as orc
from npm_policy import obs_lib as ol
from npm_policy import policy_lib as pl


def box_model(seed=0):
    # 128 points scattered over the 6 faces of a unit cube, with outward normals.
    rng = np.random.RandomState(seed)
    face = rng.randint(0, 6, size=ol.N_POINTS)
    uv = rng.uniform(-1.0, 1.0, size=(ol.N_POINTS, 2))
    pts = np.zeros((ol.N_POINTS, 3))
    nrm = np.zeros((ol.N_POINTS, 3))
    for i, f in enumerate(face):
        axis, sign = f // 2, 1.0 if f % 2 == 0 else -1.0
        other = [a for a in range(3) if a != axis]
        pts[i, axis] = sign
        pts[i, other] = uv[i]
        nrm[i, axis] = sign
    return pts, nrm, 0.25, np.array([0.0, 0.0, 0.05])


def main():
    pts, nrm, scale, centroid = box_model()
    pos = np.array([1.2, -0.4, 0.15])
    rot = Rotation.from_euler("z", 37.0, degrees=True)

    raw = orc.oracle_raw_action(rot, pos, pts, nrm, scale, centroid)
    loc = int(raw[0])
    assert raw.shape == (4,), raw.shape
    assert 0 <= loc < ol.N_POINTS, loc

    # raw[1:4] are pre-squash cone coordinates now, not a force, so the only way
    # to check them is to decode exactly as the node does.
    _loc, force_body, _pt, n_body, prev_action = pl.postprocess(
        raw, rot, pos, pts, nrm, scale, centroid)
    assert _loc == loc

    # The sigmoid cannot reach its asymptote, so a request for FORCE_MAX lands a
    # few 1e-5 N short.
    mag = float(np.linalg.norm(force_body))
    assert abs(mag - orc.ORACLE_FORCE_MAG) < 1e-3, mag

    # Strictly into the surface, and inside the cone.
    assert float(force_body @ n_body) < 0.0, float(force_body @ n_body)
    ang = np.degrees(np.arccos(np.clip((force_body / mag) @ -n_body, -1.0, 1.0)))
    assert ang <= pl.CONE_THETA_MAX_DEG + 1e-9, ang

    # prev_action carries the raw index plus the DECODED values.
    assert prev_action[0] == raw[0]
    assert abs(prev_action[1] - mag / pl.FORCE_MAX) < 1e-12
    assert float(np.linalg.norm(prev_action[2:4])) < 1.0

    # The chosen point must face away from the goal (the world origin).
    R = rot.as_matrix()
    dir_away = np.array([pos[0], pos[1], 0.0])
    dir_away /= np.linalg.norm(dir_away)
    normals_world = nrm @ R.T
    assert float(normals_world[loc] @ dir_away) > 0.0, normals_world[loc]

    # ...and be the tallest of the 30 best-scoring candidates.
    pts_world = ol.link_frame_cloud(pts, scale, centroid) @ R.T + pos
    best = np.argsort(-(normals_world @ dir_away), kind="stable")[:orc.TOPK_NORMAL]
    assert loc == int(best[np.argmax(pts_world[best, 2])]), loc

    # The decoded force is the cone-clamped version of the goal-ward, tilted-up
    # direction the oracle asked for: same angle if that fitted, else the bound.
    desired = -pos / np.linalg.norm(pos)
    desired[2] = orc.FORCE_DIR_Z
    desired /= np.linalg.norm(desired)
    want = np.degrees(np.arccos(np.clip((R.T @ desired) @ -n_body, -1.0, 1.0)))
    # U_MAX keeps |u| strictly inside the unit disk, so a clamped direction lands
    # just inside the cone rather than exactly on it.
    cap = np.degrees(np.arctan(orc.U_MAX * pl.CONE_TAN_THETA_MAX))
    assert abs(ang - min(want, cap)) < 1e-6, (ang, want, cap)

    # The push still drives the object back toward the goal, and still tilts up.
    force_world = R @ force_body
    assert float(force_world[:2] @ dir_away[:2]) < 0.0, force_world
    assert force_world[2] > 0.0, force_world

    # Deterministic: no sampling was ported.
    again = orc.oracle_raw_action(rot, pos, pts, nrm, scale, centroid)
    assert np.array_equal(raw, again)

    # Rotating object and goal-relative position together leaves the body-frame
    # action untouched. The cone anchors rotate with it: up_body is unchanged by a
    # yaw, and goal_body is R.T @ (R_extra @ goal_world), so both cancel.
    extra = Rotation.from_euler("z", 60.0, degrees=True)
    raw2 = orc.oracle_raw_action(extra * rot, extra.apply(pos), pts, nrm, scale,
                                 centroid)
    assert int(raw2[0]) == loc, (int(raw2[0]), loc)
    assert np.allclose(raw2[1:4], raw[1:4], atol=1e-9), (raw2[1:4], raw[1:4])

    print("oracle_lib OK: loc=%d |F|=%.2fN cone_angle=%.1f/%.0f deg dir_world=%s"
          % (loc, mag, ang, pl.CONE_THETA_MAX_DEG,
             np.round(force_world / np.linalg.norm(force_world), 3)))


if __name__ == "__main__":
    main()

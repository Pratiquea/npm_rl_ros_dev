#!/usr/bin/env python3
"""
Offline test of the full state -> observation -> push chain that /npm/infer
serves (no ROS). Loads the real checkpoint + npz.

Example:
    /usr/bin/python3 test_infer_pipeline.py     (override CKPT/NPZ via env vars)
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from npm_policy import policy_lib as pl
from npm_policy import obs_lib as ol

GITS = "/home/rwl-4090/gits/nonprehensile_object_manipulation"
CKPT = os.environ.get(
    "CKPT", GITS + "/logs/rsl_rl/object_manip_discrete_direct/2026-09-01_00-50-05/model_1100.pt")
NPZ = os.environ.get("NPZ", GITS + "/dataset/primitives/Paralelopiped/Paralelopiped.npz")


def test_chain(bundle, rot, pos):
    obs, raw, loc, force_body, contact_link, force_world, contact_world, next_prev = \
        pl.infer_push(bundle, rot, pos, np.zeros(3), np.zeros(3), np.zeros(4))

    assert obs.shape == (ol.OBS_LEN,), obs.shape
    assert 0 <= loc < ol.N_POINTS, loc
    fmag = float(np.linalg.norm(force_world))
    assert pl.FORCE_MIN - 1e-6 <= fmag <= pl.FORCE_MAX + 1e-6, fmag
    # Slot 769-771 of the NEXT observation. Out-of-range here means the node is
    # feeding back the pre-squash sample instead of the decoded action.
    assert pl.FORCE_MIN / pl.FORCE_MAX - 1e-9 <= next_prev[1] <= 1.0 + 1e-9, next_prev
    assert float(np.linalg.norm(next_prev[2:4])) < 1.0, next_prev
    # the contact point must sit on the object, not at the goal or the origin
    assert np.linalg.norm(contact_world - pos) < 3.0 * bundle.scale, contact_world
    # the observation must describe the state we passed in
    e = obs[ol.EXTRAS_OFF:]
    assert np.allclose(e[ol.E_OBJECT_POSE:ol.E_OBJECT_POSE + 3], pos, atol=1e-5)
    assert np.allclose(e[ol.E_OBJECT_POSE + 3:ol.E_OBJECT_POSE + 7],
                       ol.quat_wxyz(rot), atol=1e-5)
    return raw, loc, force_world, contact_world


def main():
    bundle = pl.PolicyBundle(CKPT, NPZ, mass=20.0, friction=0.95)
    print("loaded ckpt + npz, scale=%.4f" % bundle.scale)

    pos = np.array([0.8, 0.3, 0.1])
    rot = Rotation.from_euler("z", 30.0, degrees=True)
    raw, loc, fw, cw = test_chain(bundle, rot, pos)
    print("PASS chain: loc=%d |f|=%.1fN contact=%s" %
          (loc, float(np.linalg.norm(fw)), np.round(cw, 3)))

    # Determinism: act_inference is the argmax/mean, so identical state in,
    # identical action out. The coordinator relies on this per macro-step.
    raw2, loc2, _fw2, _cw2 = test_chain(bundle, rot, pos)
    assert loc2 == loc and np.allclose(raw2, raw, atol=1e-6)
    print("PASS deterministic for a repeated state")

    # The contact point must follow the object: same orientation, moved object.
    moved = pos + np.array([0.5, -0.4, 0.0])
    _raw3, loc3, _fw3, cw3 = test_chain(bundle, rot, moved)
    assert np.linalg.norm(cw3 - moved) < 3.0 * bundle.scale, cw3
    print("PASS contact tracks the object (loc=%d at the moved pose)" % loc3)

    # A yawed object must produce a yawed contact point, not the same world point.
    rot90 = Rotation.from_euler("z", 120.0, degrees=True)
    _raw4, _loc4, _fw4, cw4 = test_chain(bundle, rot90, pos)
    assert not np.allclose(cw4, cw, atol=1e-3), "rotation had no effect on contact"
    print("PASS rotation moves the contact point")
    print("ALL PASS")


if __name__ == "__main__":
    main()

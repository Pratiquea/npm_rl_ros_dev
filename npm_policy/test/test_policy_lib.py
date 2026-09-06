#!/usr/bin/env python3
"""
Offline invariant test for policy_lib (no ROS). Loads the real checkpoint + npz,
builds a realistic observation via obs_lib and asserts the deploy-path invariants.

Example:
    /usr/bin/python3 test_policy_lib.py      (override CKPT/NPZ via env vars)
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


def main():
    import torch
    pts, nrm, scale, centroid = ol.load_model_npz(NPZ)
    net, mean, std = pl.load_policy(CKPT)
    print("loaded ckpt + npz, scale=%.4f, centroid=%s" % (scale, np.round(centroid, 4)))

    # realistic pose: object 0.85 m from goal, slight yaw
    pos = np.array([0.8, 0.3, 0.1])
    rot = Rotation.from_euler("z", 30.0, degrees=True)
    obs = ol.assemble_observation(rot, pos, np.zeros(3), np.zeros(3), np.zeros(4),
                                  20.0, 0.95, scale, pts, nrm)
    assert obs.shape[0] == ol.OBS_LEN

    # empirical normalization must actually transform the obs
    xn = (torch.as_tensor(obs) - mean) / torch.clamp(std, min=1e-8)
    assert not np.allclose(xn.numpy(), obs, atol=1e-2), "normalization had no effect"

    raw = pl.infer_raw_action(net, mean, std, obs)
    assert raw.shape == (4,), raw.shape

    loc, force_body, contact_link, n_body, prev_action = pl.postprocess(
        raw, rot, pos, pts, nrm, scale, centroid)
    assert 0 <= loc < 128, loc
    # The contact point is the link-frame cloud at loc: same points the debug
    # viewers draw, so a picked point lands where rviz shows it.
    assert np.allclose(contact_link, ol.link_frame_cloud(pts, scale, centroid)[loc],
                       atol=1e-12), contact_link
    assert np.allclose(contact_link, pts[loc] * scale + centroid, atol=1e-12)
    fmag = float(np.linalg.norm(force_body))
    assert pl.FORCE_MIN - 1e-6 <= fmag <= pl.FORCE_MAX + 1e-6, fmag
    # The cone is a closed subset of the open hemisphere, so the push is strictly
    # into the surface and its angle from the inward normal is bounded.
    assert float(np.dot(force_body, n_body)) < 0.0, float(np.dot(force_body, n_body))
    ang = np.degrees(np.arccos(np.clip((force_body / fmag) @ -n_body, -1.0, 1.0)))
    assert ang <= pl.CONE_THETA_MAX_DEG + 1e-9, ang

    # prev_action is the raw index plus the DECODED magnitude and tangent coords.
    assert prev_action.shape == (4,), prev_action.shape
    assert prev_action[0] == raw[0]
    assert abs(prev_action[1] - fmag / pl.FORCE_MAX) < 1e-12
    assert float(np.linalg.norm(prev_action[2:4])) < 1.0

    force_world, contact_world = pl.to_world(force_body, contact_link, pos, rot)
    # rotation preserves magnitude and the (in)to-surface sign
    assert abs(np.linalg.norm(force_world) - np.linalg.norm(force_body)) < 1e-6
    assert float(np.dot(force_world, rot.apply(n_body))) < 0.0
    assert np.all(np.isfinite(force_world)) and np.all(np.isfinite(contact_world))

    print("PASS: loc=%d |force|=%.2fN cone_angle=%.1f/%.0f deg contact_world=%s" % (
        loc, float(np.linalg.norm(force_world)), ang, pl.CONE_THETA_MAX_DEG,
        np.round(contact_world, 3)))

    test_shrink_is_deploy_only(net, mean, std, pts, nrm, scale, centroid, rot, pos)
    test_bundle_cloud_pair(scale, centroid, pts)


def test_bundle_cloud_pair(scale, centroid, pts):
    """PolicyBundle carries both clouds; only the shrunk one may drive a push."""
    bundle = pl.PolicyBundle("", NPZ, 10.0, 0.95, load_net=False, shrink=0.97)
    assert np.allclose(bundle.link_pts_nominal,
                       ol.link_frame_cloud(pts, scale, centroid), atol=1e-12)
    assert np.allclose(bundle.link_pts,
                       ol.link_frame_cloud(pts, scale, centroid, 0.97), atol=1e-12)
    # policy_node draws obs_cloud from the nominal one and resolves loc_idx from
    # the shrunk one, so the two must not be the same array.
    assert not np.allclose(bundle.link_pts, bundle.link_pts_nominal, atol=1e-6)
    print("PASS PolicyBundle exposes the nominal cloud without moving link_pts")


def test_shrink_is_deploy_only(net, mean, std, pts, nrm, scale, centroid, rot, pos):
    """pcl_shrink moves the contact point and nothing the network sees."""
    shrink = 0.97
    bundle_obs = ol.assemble_observation(rot, pos, np.zeros(3), np.zeros(3),
                                         np.zeros(4), 20.0, 0.95, scale, pts, nrm)
    raw = pl.infer_raw_action(net, mean, std, bundle_obs)

    loc_n, _f, contact_n, _n, _p = pl.postprocess(
        raw, rot, pos, pts, nrm, scale, centroid)
    loc_s, _f, contact_s, _n, _p = pl.postprocess(
        raw, rot, pos, pts, nrm, scale, centroid, shrink)

    # Same index, different metres: the shrink is a geometry change only.
    assert loc_s == loc_n, (loc_s, loc_n)
    assert np.allclose(contact_s - centroid, shrink * (contact_n - centroid),
                       atol=1e-12)

    # The observation the net consumed is untouched: the pc block is unscaled and
    # the object_scale slot stays nominal, which is what the policy trained on.
    assert abs(float(bundle_obs[ol.EXTRAS_OFF + ol.E_OBJECT_SCALE]) - scale) < 1e-6
    assert np.allclose(bundle_obs[ol.PC_OFF:ol.NORMALS_OFF],
                       ol.rotate_cloud(rot, pts), atol=1e-6)
    print("PASS: pcl_shrink=%.2f moves the contact point %.1f mm and leaves the "
          "observation identical" % (shrink,
                                     1000 * np.linalg.norm(contact_n - contact_s)))


if __name__ == "__main__":
    main()

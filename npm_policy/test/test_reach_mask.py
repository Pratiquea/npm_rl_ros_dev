#!/usr/bin/env python3
"""
Offline test for the deploy reachability gate (no ROS). Loads the real
checkpoint + npz and asserts the properties the gate is relied on for:
the mask matches the sim's criteria, the argmax never returns a point below the
ground clearance, the force the policy produces is unchanged, and nothing on the
path can produce a NaN.

Example:
    /usr/bin/python3 test_reach_mask.py      (override CKPT/NPZ via env vars)
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


def resting_pose(link_pts, rot, xy=(0.8, 0.3)):
    """A pose that puts the object's lowest point exactly on the floor.

    The bag's failures were all floor-level picks, so the test has to reproduce
    the case where an underside exists rather than a pose floating in free space.
    """
    z = np.asarray(rot.apply(link_pts), dtype=np.float64)[:, 2]
    return np.array([xy[0], xy[1], -float(z.min())])


def main():
    import torch
    pts, nrm, scale, centroid = ol.load_model_npz(NPZ)
    link_pts = ol.link_frame_cloud(pts, scale, centroid)
    net, mean, std = pl.load_policy(CKPT)

    rot = Rotation.from_euler("z", 30.0, degrees=True)
    pos = resting_pose(link_pts, rot)
    obs = ol.assemble_observation(rot, pos, np.zeros(3), np.zeros(3),
                                  np.zeros(4), 10.0, 0.95, scale, pts, nrm)

    reach = pl.reachable_mask(rot, pos, link_pts, nrm)
    assert reach.shape == (ol.N_POINTS,), reach.shape
    assert reach.dtype == np.bool_, reach.dtype

    # 1. The mask is exactly the sim's two criteria, recomputed independently.
    R = rot.as_matrix()
    pts_world = link_pts @ R.T + pos
    nrm_world = nrm @ R.T
    expect = (pts_world[:, 2] > pl.GROUND_CLEARANCE) & \
             (nrm_world[:, 2] > pl.REACHABLE_NORMAL_Z_MIN)
    assert np.array_equal(reach, expect), "mask disagrees with the sim criteria"

    # The pose is on the floor, so both classes must be represented or the rest
    # of this test proves nothing.
    n_reach = int(reach.sum())
    assert 0 < n_reach < ol.N_POINTS, \
        "resting pose gave %d/%d reachable; test is not exercising the gate" \
        % (n_reach, ol.N_POINTS)
    low = np.where(pts_world[:, 2] <= pl.GROUND_CLEARANCE)[0]
    assert low.size > 0 and not reach[low].any()

    # 2. The masked argmax never returns an unreachable point.
    raw_m, got_n, loc_unmasked = pl.infer_raw_action_masked(
        net, mean, std, obs, reach)
    loc_m = int(raw_m[0])
    assert got_n == n_reach, (got_n, n_reach)
    assert reach[loc_m], "argmax returned unreachable loc=%d" % loc_m
    assert loc_m not in low
    assert pts_world[loc_m, 2] > pl.GROUND_CLEARANCE, pts_world[loc_m, 2]

    # 3. The force is untouched. The force head reads the soft gather over the
    # UNMASKED probs, so swapping the index must not perturb it by one bit.
    raw_u = pl.infer_raw_action(net, mean, std, obs)
    assert int(raw_u[0]) == loc_unmasked, (int(raw_u[0]), loc_unmasked)
    assert np.array_equal(np.float64(raw_u[1:4]), np.float64(raw_m[1:4])), \
        "masking perturbed the continuous action: %s vs %s" % (raw_u[1:4], raw_m[1:4])

    # 4. Nothing on the path can go non-finite. This is the -inf failure mode the
    # gate deliberately avoids by subtracting a finite sentinel instead.
    assert np.all(np.isfinite(raw_m)), raw_m
    xn = (torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
          - mean) / torch.clamp(std, min=1e-8)
    with torch.no_grad():
        cat, _g = net.update_distribution(xn)
        scores = cat.logits.detach().clone().reshape(-1)
    assert bool(torch.isfinite(scores).all()), "raw logits already non-finite"
    # Kept before the in-place penalty below, for the sampling checks in 9.
    raw_scores = np.asarray(scores.cpu(), dtype=np.float64)
    scores[~torch.as_tensor(reach)] -= pl.UNREACHABLE_PENALTY
    assert bool(torch.isfinite(scores).all()), "masked scores went non-finite"
    # The sentinel has to dominate the logit range, or a high-scoring unreachable
    # point could still win.
    assert float(scores.max() - scores.min()) < 2.0 * pl.UNREACHABLE_PENALTY

    # 5. All reachable is a no-op: the gate must not change behaviour when it has
    # nothing to reject.
    raw_all, n_all, _u = pl.infer_raw_action_masked(
        net, mean, std, obs, np.ones(ol.N_POINTS, dtype=bool))
    assert n_all == ol.N_POINTS
    assert int(raw_all[0]) == int(raw_u[0]), (int(raw_all[0]), int(raw_u[0]))

    # 6. All unreachable degrades to the unmasked choice instead of failing.
    # Every score takes the same penalty, so the ordering survives.
    raw_none, n_none, _u = pl.infer_raw_action_masked(
        net, mean, std, obs, np.zeros(ol.N_POINTS, dtype=bool))
    assert n_none == 0
    assert int(raw_none[0]) == int(raw_u[0]), (int(raw_none[0]), int(raw_u[0]))
    assert np.all(np.isfinite(raw_none)), raw_none

    # 7. A wrong-length mask is a bug, not something to silently broadcast over.
    try:
        pl.infer_raw_action_masked(net, mean, std, obs,
                                   np.ones(ol.N_POINTS - 1, dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("short mask was accepted")

    # 8. infer_push threads the gate end to end and its contact point clears the
    # floor, which is the property the executor and the operator depend on.
    bundle = pl.PolicyBundle(CKPT, NPZ, 10.0, 0.95)
    _o, _r, loc_p, _fb, _cl, _fw, contact_world, _pa = pl.infer_push(
        bundle, rot, pos, np.zeros(3), np.zeros(3), np.zeros(4), mask_reach=True)
    assert reach[loc_p], loc_p
    assert contact_world[2] > pl.GROUND_CLEARANCE, contact_world[2]
    # The gate and to_world must agree on where the point IS, or the mask could
    # be testing one geometry while the executor is handed another. to_world is
    # an independent path (rot.apply on the link cloud), so this is not circular.
    assert np.allclose(contact_world, pts_world[loc_p], atol=1e-9), \
        (contact_world, pts_world[loc_p])

    # 9. Sampling. top_k>1 must widen the choice without ever leaving the
    # reachable set, and without touching the force the policy asked for.
    raw_k1, _n1, _u1 = pl.infer_raw_action_masked(
        net, mean, std, obs, reach, top_k=1)
    assert int(raw_k1[0]) == loc_m, (int(raw_k1[0]), loc_m)

    masked_scores = raw_scores - pl.UNREACHABLE_PENALTY * (~reach)
    top3 = set(int(i) for i in np.argsort(-masked_scores, kind="stable")[:3])
    assert loc_m in top3

    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        raw_s, n_s, u_s = pl.infer_raw_action_masked(
            net, mean, std, obs, reach, top_k=3, rng=rng)
        loc_s = int(raw_s[0])
        assert reach[loc_s], "sampled unreachable loc=%d" % loc_s
        assert loc_s in top3, (loc_s, sorted(top3))
        assert n_s == n_reach and u_s == loc_unmasked
        assert np.array_equal(np.float64(raw_u[1:4]), np.float64(raw_s[1:4])), \
            "sampling perturbed the continuous action"
        assert np.all(np.isfinite(raw_s)), raw_s
        seen.add(loc_s)
    assert len(seen) > 1, "top_k=3 never sampled anything but %s" % seen

    # A fixed seed is reproducible, or a bag cannot be replayed.
    a = [int(pl.infer_raw_action_masked(net, mean, std, obs, reach, top_k=3,
                                        rng=np.random.default_rng(7))[0][0])
         for _ in range(5)]
    assert len(set(a)) == 1, a

    # top_k larger than the reachable pool must not reach into the penalized
    # block: capping at n_reachable is the property, not clipping at N.
    for _ in range(100):
        loc_b, _u, cand = pl.select_loc(raw_scores, reach,
                                        top_k=ol.N_POINTS, rng=rng)
        assert cand.size == n_reach, (cand.size, n_reach)
        assert reach[loc_b], loc_b

    # All-unreachable degrades to the single unmasked argmax, sampling or not.
    loc_z, _u, cand_z = pl.select_loc(raw_scores,
                                      np.zeros(ol.N_POINTS, dtype=bool),
                                      top_k=3, rng=rng)
    assert cand_z.size == 1 and loc_z == loc_unmasked, (loc_z, loc_unmasked)

    # min_sep: candidates must be that far apart in world space, which is the
    # whole point (the top logits are otherwise neighbours on one face).
    MIN_SEP = 0.05
    _loc, _u, cand_sep = pl.select_loc(raw_scores, reach, top_k=3, rng=rng,
                                       min_sep=MIN_SEP, pts_world=pts_world)
    d = np.linalg.norm(pts_world[cand_sep][:, None, :]
                       - pts_world[cand_sep][None, :, :], axis=-1)
    d[np.diag_indices_from(d)] = np.inf
    assert d.min() >= MIN_SEP - 1e-9, (d.min(), MIN_SEP)
    assert int(cand_sep[0]) == loc_m, "spread dropped the best point"

    # min_sep without the cloud is a bug, not a silent fallback to no spread.
    try:
        pl.select_loc(raw_scores, reach, top_k=3, min_sep=MIN_SEP)
    except ValueError:
        pass
    else:
        raise AssertionError("min_sep without pts_world was accepted")

    # reachable_mask's own world cloud must be the one the spread uses.
    mask_rw, pts_rw = pl.reachable_mask(rot, pos, link_pts, nrm,
                                        return_world=True)
    assert np.array_equal(mask_rw, reach)
    assert np.allclose(pts_rw, pts_world, atol=1e-12)

    # infer_push threads sampling end to end and still clears the floor.
    for _ in range(20):
        _o, _r, loc_ip, _fb, _cl, _fw, cw_ip, _pa = pl.infer_push(
            bundle, rot, pos, np.zeros(3), np.zeros(3), np.zeros(4),
            mask_reach=True, top_k=3, rng=rng, min_sep=MIN_SEP)
        assert reach[loc_ip], loc_ip
        assert cw_ip[2] > pl.GROUND_CLEARANCE, cw_ip[2]

    print("PASS: reach=%d/%d unmasked_loc=%d masked_loc=%d z=%.4f "
          "(moved=%s) infer_push loc=%d z=%.4f | top3=%s sampled=%s spread=%s"
          % (n_reach, ol.N_POINTS, loc_unmasked, loc_m, pts_world[loc_m, 2],
             loc_unmasked != loc_m, loc_p, contact_world[2],
             sorted(top3), sorted(seen), list(cand_sep)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import math

import numpy as np

from npm_policy.obs_lib import (N_POINTS, OBS_LEN, assemble_observation,  # noqa: F401
                                link_frame_cloud, load_model_npz)

CONE_THETA_MAX_DEG = 20.0
CONE_TAN_THETA_MAX = math.tan(math.radians(CONE_THETA_MAX_DEG))
CONE_FRAME_BLEND_COS_LO = 0.70
CONE_FRAME_BLEND_COS_HI = 0.98
FORCE_MIN = 10.0
FORCE_MAX = 75.0

GROUND_CLEARANCE = 0.04
REACHABLE_NORMAL_Z_MIN = -0.3

UNREACHABLE_PENALTY = 1e4

DEFAULT_CFG = dict(
    pointnet_hidden=[256, 128], embed_dim=64, query_hidden_dims=[128],
    critic_hidden_dims=[32, 32], actor_hidden_dims=[32, 32], activation="elu",
    pool_mode="mean+max", init_noise_std=1.0, state_dependent_std=False,
    log_std_min=-5.0, log_std_max=2.0,
)


def load_policy(checkpoint_path, cfg=None, device="cpu"):
    import torch
    from npm_policy.actor_critic_smdp import ActorCriticPointerCategoricalSMDP
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    net = ActorCriticPointerCategoricalSMDP(
        num_actor_obs=OBS_LEN, num_critic_obs=OBS_LEN, num_actions=4,
        num_points=N_POINTS, point_dim=3, **cfg)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(sd["model_state_dict"])
    net.eval()
    on = sd["obs_norm_state_dict"]
    mean = on["_mean"].to(device).float().reshape(-1)
    std = on["_std"].to(device).float().reshape(-1)
    if mean.numel() != OBS_LEN or std.numel() != OBS_LEN:
        raise ValueError(
            "checkpoint %s has obs norm stats of length %d/%d, expected %d: "
            "it was trained on a different observation layout"
            % (checkpoint_path, mean.numel(), std.numel(), OBS_LEN))
    return net, mean, std


def infer_raw_action(net, mean, std, obs798, device="cpu"):
    import torch
    x = torch.as_tensor(np.asarray(obs798, dtype=np.float32), device=device).reshape(1, -1)
    xn = (x - mean) / torch.clamp(std, min=1e-8)
    with torch.no_grad():
        a = net.act_inference(xn)
    return a.cpu().numpy().reshape(-1)


def reachable_mask(rot, pos_world, link_pts, normals, floor_z=0.0,
                   ground_clearance=GROUND_CLEARANCE,
                   normal_z_min=REACHABLE_NORMAL_Z_MIN, return_world=False):
    R = np.asarray(rot.as_matrix(), dtype=np.float64)
    pts = np.asarray(link_pts, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(pos_world, dtype=np.float64).reshape(3)

    pts_world = pts @ R.T + pos
    normals_world = nrm @ R.T

    above_ground = pts_world[:, 2] > (float(floor_z) + float(ground_clearance))
    normal_ok = normals_world[:, 2] > float(normal_z_min)
    mask = above_ground & normal_ok
    return (mask, pts_world) if return_world else mask


def _spread_candidates(pool, k, min_sep, pts_world):
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    kept = []
    for i in pool:
        if len(kept) >= k:
            break
        if not kept or np.all(np.linalg.norm(pts[kept] - pts[i], axis=1) >= min_sep):
            kept.append(int(i))
    if len(kept) < k:
        for i in pool:
            if len(kept) >= k:
                break
            if int(i) not in kept:
                kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def select_loc(scores, mask, top_k=1, rng=None, min_sep=0.0, pts_world=None):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if scores.shape != m.shape:
        raise ValueError(
            "reachable mask has %d entries but the pointer has %d logits"
            % (m.size, scores.size))
    loc_unmasked = int(np.argmax(scores))

    masked = scores - UNREACHABLE_PENALTY * (~m)
    order = np.argsort(-masked, kind="stable")
    n_reach = int(m.sum())
    k = int(top_k)
    if k <= 1 or n_reach == 0:
        return int(order[0]), loc_unmasked, order[:1].astype(np.int64)

    pool = order[:n_reach]
    k = min(k, int(pool.size))
    if min_sep > 0.0:
        if pts_world is None:
            raise ValueError("min_sep=%g requires pts_world" % min_sep)
        cand = _spread_candidates(pool, k, float(min_sep), pts_world)
    else:
        cand = pool[:k].astype(np.int64)
    if rng is None:
        rng = np.random.default_rng()
    return int(cand[int(rng.integers(cand.size))]), loc_unmasked, cand


def infer_raw_action_masked(net, mean, std, obs798, reachable, device="cpu",
                            top_k=1, rng=None, min_sep=0.0, pts_world=None):
    import torch
    x = torch.as_tensor(np.asarray(obs798, dtype=np.float32),
                        device=device).reshape(1, -1)
    xn = (x - mean) / torch.clamp(std, min=1e-8)
    m = np.asarray(reachable, dtype=bool).reshape(-1)
    with torch.no_grad():
        cat, gauss = net.update_distribution(xn)
        scores = cat.logits.detach().reshape(-1).cpu().numpy()
        cont = gauss.mean.detach().reshape(-1).cpu().numpy()
    loc, loc_unmasked, _cand = select_loc(scores, m, top_k=top_k, rng=rng,
                                          min_sep=min_sep, pts_world=pts_world)
    return np.concatenate([[float(loc)], cont]), int(m.sum()), loc_unmasked


def _sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def cone_frame_anchors(rot, pos_world):
    R = np.asarray(rot.as_matrix(), dtype=np.float64)
    up_body = R[2, :].copy()

    pos = np.asarray(pos_world, dtype=np.float64).reshape(3)
    xy = pos[:2]
    dist = float(np.linalg.norm(xy))
    goal_world = np.zeros(3, dtype=np.float64)
    goal_world[:2] = -xy / max(dist, 1e-6)
    if dist <= 1e-4:
        goal_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    goal_body = R.T @ goal_world
    return up_body, goal_body


def tangent_basis_anchored(n, up, fallback, cos_lo=CONE_FRAME_BLEND_COS_LO,
                           cos_hi=CONE_FRAME_BLEND_COS_HI, eps=1e-8):
    n = np.asarray(n, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    fallback = np.asarray(fallback, dtype=np.float64).reshape(3)

    c = abs(float(np.dot(n, up)))
    w = float(np.clip((c - cos_lo) / max(cos_hi - cos_lo, eps), 0.0, 1.0))

    anchor = (1.0 - w) * up + w * fallback
    anchor = anchor / max(float(np.linalg.norm(anchor)), eps)

    t1 = np.cross(anchor, n)
    t1 = t1 / max(float(np.linalg.norm(t1)), eps)
    t2 = np.cross(n, t1)
    return t1, t2


def decode_cone_action(a_m, a_tan, n_hat, up_body, goal_body):
    n_hat = np.asarray(n_hat, dtype=np.float64).reshape(3)
    t1, t2 = tangent_basis_anchored(n_hat, up_body, goal_body)

    a = np.asarray(a_tan, dtype=np.float64).reshape(2)
    u = a / math.sqrt(1.0 + float(np.dot(a, a)))

    d = -n_hat + CONE_TAN_THETA_MAX * (u[0] * t1 + u[1] * t2)
    d = d / max(float(np.linalg.norm(d)), 1e-6)

    force_mag = FORCE_MIN + (FORCE_MAX - FORCE_MIN) * _sigmoid(float(a_m))
    force_body = force_mag * d
    decoded = np.array([force_mag / FORCE_MAX, u[0], u[1]], dtype=np.float64)
    return force_body, decoded


TAPER_P_CAP = 0.95


def taper_gain(dist, taper_dist):
    return float(np.clip(float(dist) / max(float(taper_dist), 1e-6), 0.0, 1.0))


def taper_prob(dist, taper_dist, p_far, p_near, p_cap=TAPER_P_CAP):
    g = taper_gain(dist, taper_dist)
    p = float(p_near) + (float(p_far) - float(p_near)) * g
    return float(np.clip(p, 0.0, min(float(p_cap), 1.0)))


def apply_force_taper(force_body, dist, taper_dist, taper_force_min, p_far, p_near,
                      rng=None):
    f = np.asarray(force_body, dtype=np.float64).reshape(3)
    mag = float(np.linalg.norm(f))
    if mag <= 1e-9:
        return f
    if rng is None:
        rng = np.random.default_rng()
    if float(rng.random()) >= taper_prob(dist, taper_dist, p_far, p_near):
        return f
    mag_t = float(np.clip(mag * taper_gain(dist, taper_dist),
                          float(taper_force_min), mag))
    return f * (mag_t / mag)


def postprocess(raw_action, rot, pos_world, pts_norm, normals, scale, centroid,
                shrink=1.0):
    raw = np.asarray(raw_action, dtype=np.float64).reshape(4)
    loc = int(np.clip(np.trunc(raw[0]), 0, N_POINTS - 1))

    normal_body = np.asarray(normals[loc], dtype=np.float64)
    up_body, goal_body = cone_frame_anchors(rot, pos_world)
    force_body, decoded = decode_cone_action(raw[1], raw[2:4], normal_body,
                                             up_body, goal_body)
    contact_link = link_frame_cloud(pts_norm, scale, centroid, shrink)[loc]

    prev_action = np.concatenate([[raw[0]], decoded])
    return loc, force_body, contact_link, normal_body, prev_action


class PolicyBundle(object):

    def __init__(self, checkpoint_path, npz_path, mass, friction, device="cpu",
                 load_net=True, shrink=1.0):
        if load_net:
            self.net, self.mean, self.std = load_policy(checkpoint_path, device=device)
        else:
            self.net, self.mean, self.std = None, None, None
        self.pts, self.normals, self.scale, self.centroid = load_model_npz(npz_path)
        self.shrink = float(shrink)
        self.link_pts = link_frame_cloud(self.pts, self.scale, self.centroid,
                                         self.shrink)
        self.link_pts_nominal = link_frame_cloud(self.pts, self.scale, self.centroid)
        self.mass = float(mass)
        self.friction = float(friction)
        self.device = device


def infer_push(bundle, rot, pos_world, lin_vel, ang_vel, prev_action,
               mask_reach=False, floor_z=0.0,
               ground_clearance=GROUND_CLEARANCE,
               normal_z_min=REACHABLE_NORMAL_Z_MIN,
               top_k=1, rng=None, min_sep=0.0):
    obs = assemble_observation(rot, pos_world, lin_vel, ang_vel, prev_action,
                               bundle.mass, bundle.friction, bundle.scale,
                               bundle.pts, bundle.normals)
    if mask_reach:
        reach, pts_world = reachable_mask(
            rot, pos_world, bundle.link_pts, bundle.normals, floor_z=floor_z,
            ground_clearance=ground_clearance, normal_z_min=normal_z_min,
            return_world=True)
        raw, _n_reach, _loc_unmasked = infer_raw_action_masked(
            bundle.net, bundle.mean, bundle.std, obs, reach, bundle.device,
            top_k=top_k, rng=rng, min_sep=min_sep, pts_world=pts_world)
    else:
        raw = infer_raw_action(bundle.net, bundle.mean, bundle.std, obs,
                               bundle.device)
    loc, force_body, contact_link, _n, next_prev = postprocess(
        raw, rot, pos_world, bundle.pts, bundle.normals, bundle.scale,
        bundle.centroid, bundle.shrink)
    force_world, contact_world = to_world(force_body, contact_link, pos_world, rot)
    return (obs, raw, loc, force_body, contact_link, force_world, contact_world,
            next_prev)


def to_world(force_body, contact_link, pos_world, rot):
    force_world = np.asarray(rot.apply(np.asarray(force_body, dtype=np.float64)))
    contact_world = np.asarray(rot.apply(np.asarray(contact_link, dtype=np.float64))) \
        + np.asarray(pos_world, dtype=np.float64)
    return force_world, contact_world

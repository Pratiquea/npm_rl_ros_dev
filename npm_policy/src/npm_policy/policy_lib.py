#!/usr/bin/env python3
"""Policy load, inference, cone action decode, and body->world frame conversion.

Mirrors the deploy path of objectmanip_env_discrete.step(): empirical
normalization -> act_inference -> _decode_cone_action -> rotate into world.

No ROS imports, so it is unit-testable offline. torch is imported lazily by the
two functions that run the net, so the oracle path works without torch installed.
"""
import math

import numpy as np

from npm_policy.obs_lib import (N_POINTS, OBS_LEN, assemble_observation,  # noqa: F401
                                link_frame_cloud, load_model_npz)

# Cone action parametrization (ObjectManipDiscreteEnvCfg). The net's continuous
# outputs are pre-squash coordinates, not a force: magnitude comes from a sigmoid
# and direction from a tangent-disk map bounded to CONE_THETA_MAX_DEG about the
# inward normal. There is no force scale and no norm clip any more; the cone map
# is exactly bounded by construction.
CONE_THETA_MAX_DEG = 20.0
CONE_TAN_THETA_MAX = math.tan(math.radians(CONE_THETA_MAX_DEG))
CONE_FRAME_BLEND_COS_LO = 0.70
CONE_FRAME_BLEND_COS_HI = 0.98
FORCE_MIN = 10.0
FORCE_MAX = 75.0

# Deploy-only reachability gate (objectmanip_env_discrete.py:144-145). The sim
# computes the same mask in _compute_reachable_mask but never enforces it: both
# the logits mask (sim:1240) and the force zeroing (sim:1478) are commented out,
# leaving only the -0.3 unreachable_pick_penalty to shape it away in training.
# That is enough in sim because an unreachable pick still gets its full force, so
# the object moves and the next observation differs. On the robot the push cannot
# happen at all, the observation is frozen, and the deterministic argmax returns
# the same index forever (hand_push_policy.bag: loc_idx=5 sixteen times, 44 s, no
# object motion). The gate has to be enforced here instead.
GROUND_CLEARANCE = 0.04
REACHABLE_NORMAL_Z_MIN = -0.3

# Finite sentinel, deliberately not -inf. compute_pointer_logits bounds every
# logit to [-30, 30] (actor_critic_smdp.py:64,69), a spread of 60, so this
# dominates absolutely: a reachable point always outranks an unreachable one.
# -inf would put deploy back where training was, because a Categorical built on
# it gives a masked point p=0 and log p=-inf, and entropy() (actor_critic_smdp.py
# :328) then evaluates 0 * -inf = NaN. Subtraction and argmax over finite floats
# cannot produce NaN.
UNREACHABLE_PENALTY = 1e4

# Architecture for the object_manip_discrete checkpoint (params/agent.yaml).
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
    # strict=True: a checkpoint from a different architecture should fail
    # rather than partially load.
    net.load_state_dict(sd["model_state_dict"])
    net.eval()
    on = sd["obs_norm_state_dict"]
    mean = on["_mean"].to(device).float().reshape(-1)
    std = on["_std"].to(device).float().reshape(-1)
    # verify observation lenght/layout
    if mean.numel() != OBS_LEN or std.numel() != OBS_LEN:
        raise ValueError(
            "checkpoint %s has obs norm stats of length %d/%d, expected %d: "
            "it was trained on a different observation layout"
            % (checkpoint_path, mean.numel(), std.numel(), OBS_LEN))
    return net, mean, std


def infer_raw_action(net, mean, std, obs798, device="cpu"):
    """Normalize -> act_inference -> raw [loc_idx_float, a_m, a1, a2] (pre-squash)."""
    import torch
    x = torch.as_tensor(np.asarray(obs798, dtype=np.float32), device=device).reshape(1, -1)
    xn = (x - mean) / torch.clamp(std, min=1e-8)
    with torch.no_grad():
        a = net.act_inference(xn)
    return a.cpu().numpy().reshape(-1)


def reachable_mask(rot, pos_world, link_pts, normals, floor_z=0.0,
                   ground_clearance=GROUND_CLEARANCE,
                   normal_z_min=REACHABLE_NORMAL_Z_MIN, return_world=False):
    """(N,) bool, True where a point is a candidate contact for the real robot.

    Port of the sim's _compute_reachable_mask (objectmanip_env_discrete.py:873-905),
    same two criteria and same thresholds:
      1. the point sits high enough above the floor to get a finger behind it,
      2. its normal does not point hard downwards.

    link_pts must be the LINK-frame metric cloud (PolicyBundle.link_pts), the
    analogue of the sim's model_pcl_scaled_link_frame. rot/pos_world must be the
    same pose snapshot the observation was assembled from, for the same reason
    postprocess insists on it.

    floor_z exists because the threshold is only meaningful against the ground
    plane. The world origin is the mocap origin, set by the OptiTrack ground-plane
    calibration, so 0.0 is right for this rig and wrong the moment that changes.

    obs_lib.rotate_cloud is not reused: it flattens and downcasts to float32 for
    the observation block, and the comparisons here want (N,3).

    return_world also hands back the (N,3) world-frame cloud the criteria were
    evaluated on, so a caller doing min_sep candidate spreading does not rotate
    the same points a second time and risk spreading over a different geometry
    than the one that was gated.
    """
    R = np.asarray(rot.as_matrix(), dtype=np.float64)
    pts = np.asarray(link_pts, dtype=np.float64).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(pos_world, dtype=np.float64).reshape(3)

    pts_world = pts @ R.T + pos
    # Normals rotate but do not translate. The sim does not renormalize either
    # (sim:902), and a rotation cannot change the length anyway.
    normals_world = nrm @ R.T

    above_ground = pts_world[:, 2] > (float(floor_z) + float(ground_clearance))
    normal_ok = normals_world[:, 2] > float(normal_z_min)
    mask = above_ground & normal_ok
    return (mask, pts_world) if return_world else mask


def _spread_candidates(pool, k, min_sep, pts_world):
    """Greedy walk down `pool` keeping points at least min_sep apart, up to k.

    The pointer's top logits are usually adjacent points on the same face: on the
    Paralelopiped's 128-point cloud the runners-up sit centimetres from the
    winner, so sampling among them re-picks the same contact geometry. This keeps
    the highest-scoring point of each cluster instead.

    Falls back to filling from the head of `pool` if the separation constraint
    cannot produce k candidates, so the caller always gets min(k, pool size).
    """
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
    """Pick the pointer index from raw logits under the reachability gate.

    Returns (loc, loc_unmasked, candidates). numpy only, so it is testable
    without torch or a checkpoint.

    top_k <= 1 is the deterministic masked argmax, bit-identical to what this
    module did before sampling existed. top_k > 1 draws uniformly from the k
    highest-scoring REACHABLE points, which is what breaks the livelock where a
    contact that cannot move the object freezes the observation and the argmax
    then returns the same index forever.

    The candidate pool is capped at n_reachable, so an unreachable point is never
    sampled however large top_k is. When nothing is reachable the pool would be
    empty and every score carries the same penalty anyway, so this degrades to
    the single unmasked argmax rather than sampling from a set it just rejected.

    min_sep (m) additionally requires candidates to be that far apart in world
    space, and then pts_world is required.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if scores.shape != m.shape:
        raise ValueError(
            "reachable mask has %d entries but the pointer has %d logits"
            % (m.size, scores.size))
    loc_unmasked = int(np.argmax(scores))

    masked = scores - UNREACHABLE_PENALTY * (~m)
    # Stable sort so ties keep index order, i.e. order[0] is exactly argmax.
    order = np.argsort(-masked, kind="stable")
    n_reach = int(m.sum())
    k = int(top_k)
    if k <= 1 or n_reach == 0:
        return int(order[0]), loc_unmasked, order[:1].astype(np.int64)

    # Reachable points all outrank unreachable ones by construction, so the first
    # n_reach entries of `order` are exactly the reachable pool, best first.
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
    """infer_raw_action, but the pointer argmax skips unreachable points.

    Returns (raw_action, n_reachable, loc_unmasked). raw_action keeps the
    [loc_idx_float, a_m, a1, a2] layout, so postprocess is unchanged.
    loc_unmasked is the index the unmasked argmax would have picked; it is free
    here and it is the only way to log how far the gate moved the choice without
    paying for a second forward pass.

    update_distribution is called instead of act_inference because the mask has
    to sit between the logits and the argmax and act_inference fuses the two.
    The mask cannot go inside the net: actor_critic_smdp.py is sha256-locked to
    the sim's copy by test_net_sync.py.

    gauss.mean comes back untouched, so the continuous action is identical to the
    unmasked call. The force head is conditioned on the soft gather over the
    UNMASKED probs (actor_critic_smdp.py:233-234) and never sees the hard index,
    which is what makes substituting the index in-distribution rather than a
    hack. The force that finally leaves postprocess still differs, because the
    cone is anchored at normals[loc]; that is what the sim does for any index.

    No Categorical is built from the masked scores and no log_prob/entropy/probs
    is called on them, so the NaN that -inf masking causes in training has no
    path here.

    All-unreachable is not an error: every score takes the same penalty, the
    ordering survives, and the argmax returns the unmasked choice. The caller
    gets n_reachable == 0 and is expected to say so out loud.
    """
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
    # Branchless exp(-|x|) form: the plain 1/(1+exp(-x)) overflows for very
    # negative a_m, which the unbounded Gaussian mean can reach.
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def cone_frame_anchors(rot, pos_world):
    """Returns (up_body, goal_body): world up and the horizontal object-to-goal
    direction, both expressed in the object BODY frame, where the normals and the
    force live.

    rot: scipy Rotation, object orientation in `world`. pos_world doubles as the
    position relative to the goal, which is the world origin.
    """
    R = np.asarray(rot.as_matrix(), dtype=np.float64)      # body -> world
    # Row 2, not column 2. R[2, :] == R.T @ e_z == world up in body coords, which
    # is what anchors the tangent frame. R[:, 2] is the body z-axis in world and
    # would silently point the tilt axis somewhere else.
    up_body = R[2, :].copy()

    pos = np.asarray(pos_world, dtype=np.float64).reshape(3)
    xy = pos[:2]
    dist = float(np.linalg.norm(xy))
    goal_world = np.zeros(3, dtype=np.float64)
    goal_world[:2] = -xy / max(dist, 1e-6)
    # Sitting on the goal leaves the direction undefined; any horizontal unit
    # vector works.
    if dist <= 1e-4:
        goal_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    goal_body = R.T @ goal_world
    return up_body, goal_body


def tangent_basis_anchored(n, up, fallback, cos_lo=CONE_FRAME_BLEND_COS_LO,
                           cos_hi=CONE_FRAME_BLEND_COS_HI, eps=1e-8):
    """Tangent frame about n whose roll is pinned by an external anchor.

    A basis built from n alone has a free roll and, by the hairy-ball theorem,
    cannot be continuous over the sphere, so the same tangent coordinates would
    mean a different force at every point. Anchoring to `up` pins it: t1 comes out
    lateral and t2 upward.

    The anchor degenerates as n aligns with up, so over |n.up| in [cos_lo, cos_hi]
    it blends into `fallback` (which must be perpendicular to `up`). Inside that
    band t2 stops meaning "tilt up" and starts meaning "tilt toward the goal".

    n, up, fallback: (3,) unit vectors in one common frame.
    """
    n = np.asarray(n, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    fallback = np.asarray(fallback, dtype=np.float64).reshape(3)

    c = abs(float(np.dot(n, up)))
    w = float(np.clip((c - cos_lo) / max(cos_hi - cos_lo, eps), 0.0, 1.0))

    anchor = (1.0 - w) * up + w * fallback
    anchor = anchor / max(float(np.linalg.norm(anchor)), eps)

    # Order matters: anchor x n, not n x anchor, or the lateral axis flips sign.
    t1 = np.cross(anchor, n)
    t1 = t1 / max(float(np.linalg.norm(t1)), eps)
    t2 = np.cross(n, t1)
    return t1, t2


def decode_cone_action(a_m, a_tan, n_hat, up_body, goal_body):
    """Map the raw Gaussian action onto a force inside a cone about the inward normal.

    Returns (force_body(3,), decoded(3,)), where decoded is [|F|/FORCE_MAX, u1, u2]
    and is what the sim writes back into the next observation's prev_action.
    """
    n_hat = np.asarray(n_hat, dtype=np.float64).reshape(3)
    t1, t2 = tangent_basis_anchored(n_hat, up_body, goal_body)

    # |u| < 1 for any finite a_tan, so the direction is always cone bounded: the
    # angle from -n_hat is atan(|u| * tan(theta_max)) < theta_max. No clip, no
    # fold, and no dead zone for the policy gradient to sit in.
    a = np.asarray(a_tan, dtype=np.float64).reshape(2)
    u = a / math.sqrt(1.0 + float(np.dot(a, a)))

    d = -n_hat + CONE_TAN_THETA_MAX * (u[0] * t1 + u[1] * t2)
    d = d / max(float(np.linalg.norm(d)), 1e-6)

    force_mag = FORCE_MIN + (FORCE_MAX - FORCE_MIN) * _sigmoid(float(a_m))
    force_body = force_mag * d
    decoded = np.array([force_mag / FORCE_MAX, u[0], u[1]], dtype=np.float64)
    return force_body, decoded


def postprocess(raw_action, rot, pos_world, pts_norm, normals, scale, centroid,
                shrink=1.0):
    """Returns (loc_idx, force_body(3,), contact_link(3,), normal_body(3,), prev_action(4,)).

    rot/pos_world MUST be the same pose snapshot the observation was assembled
    from: the sim builds the cone frame from the pose cached during the previous
    _get_observations, so re-reading a fresher pose here would decode the action
    in a frame the policy never saw.
    """
    raw = np.asarray(raw_action, dtype=np.float64).reshape(4)
    # Truncation, matching the sim's action[:, 0].long(). act_inference emits an
    # exact integer, so this only guards against a hand-built action.
    loc = int(np.clip(np.trunc(raw[0]), 0, N_POINTS - 1))

    normal_body = np.asarray(normals[loc], dtype=np.float64)
    up_body, goal_body = cone_frame_anchors(rot, pos_world)
    force_body, decoded = decode_cone_action(raw[1], raw[2:4], normal_body,
                                             up_body, goal_body)
    contact_link = link_frame_cloud(pts_norm, scale, centroid, shrink)[loc]

    # The sim keeps the raw index but replaces the continuous slots with the
    # DECODED values (objectmanip_env_discrete.py:1754-1755). Feeding the
    # pre-squash sample here instead corrupts extras 769-771 of every later obs.
    prev_action = np.concatenate([[raw[0]], decoded])
    return loc, force_body, contact_link, normal_body, prev_action


class PolicyBundle(object):
    """Checkpoint + object model, everything inference needs besides the state."""

    def __init__(self, checkpoint_path, npz_path, mass, friction, device="cpu",
                 load_net=True, shrink=1.0):
        # load_net=False is the oracle path: it needs the object model but no
        # checkpoint, so the node can run without weights and no GPU.
        if load_net:
            self.net, self.mean, self.std = load_policy(checkpoint_path, device=device)
        else:
            self.net, self.mean, self.std = None, None, None
        self.pts, self.normals, self.scale, self.centroid = load_model_npz(npz_path)
        # Deploy-only contraction of the metric cloud (see link_frame_cloud). It
        # moves contact points and the reach mask, never the observation.
        self.shrink = float(shrink)
        # Metric cloud about the link origin, i.e. about the mocap pose. The
        # observation stays on unscaled, unshrunk self.pts.
        self.link_pts = link_frame_cloud(self.pts, self.scale, self.centroid,
                                         self.shrink)
        # The same cloud without the margin. Debug draw only: the contact point
        # and the reach mask must keep using the shrunk link_pts above.
        self.link_pts_nominal = link_frame_cloud(self.pts, self.scale, self.centroid)
        self.mass = float(mass)
        self.friction = float(friction)
        self.device = device


def infer_push(bundle, rot, pos_world, lin_vel, ang_vel, prev_action,
               mask_reach=False, floor_z=0.0,
               ground_clearance=GROUND_CLEARANCE,
               normal_z_min=REACHABLE_NORMAL_Z_MIN,
               top_k=1, rng=None, min_sep=0.0):
    """One macro-step of the deploy path, from object state to world-frame push.

    Returns (obs, raw_action, loc, force_body, contact_link, force_world,
    contact_world, next_prev_action). Kept out of the node so the whole chain is
    testable offline; policy_node._act is the same sequence with an oracle branch,
    so the two must be changed together.

    mask_reach defaults False so existing offline callers keep the unmasked
    argmax. The bag replay needs both, to show the unmasked path still reproduces
    what was recorded before it claims the masked path fixes it.

    top_k/rng/min_sep default to the deterministic pick for the same reason: a
    replay must stay reproducible unless it asks not to be. They only apply on
    the mask_reach path, where select_loc has a reachable pool to sample from.
    """
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
    # The sim applies force and contact point in the object BODY/LINK frame
    # (set_external_force_and_torque does no rotation), so deploy must rotate both.
    force_world = np.asarray(rot.apply(np.asarray(force_body, dtype=np.float64)))
    contact_world = np.asarray(rot.apply(np.asarray(contact_link, dtype=np.float64))) \
        + np.asarray(pos_world, dtype=np.float64)
    return force_world, contact_world

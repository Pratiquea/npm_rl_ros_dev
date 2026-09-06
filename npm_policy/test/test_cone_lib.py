#!/usr/bin/env python3
"""
Offline test of the cone action parametrization (no ROS, no checkpoint). The last
case cross-checks against the sim's own torch implementation when it is reachable.

Example:
    /usr/bin/python3 test_cone_lib.py
"""
import io
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from npm_policy import oracle_lib as orc
from npm_policy import policy_lib as pl

SIM_REPO = os.environ.get(
    "NPM_SIM_REPO", os.path.expanduser("~/gits/nonprehensile_object_manipulation"))


def rand_unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_row_vs_column():
    # A transposed R is the likeliest porting bug: it passes every magnitude and
    # cone-angle assert while pointing the tilt axis somewhere else. Under a 90 deg
    # rotation about world x, world up must land on body +y; R[:, 2] gives -y.
    rot = Rotation.from_euler("x", 90.0, degrees=True)
    up_body, _ = pl.cone_frame_anchors(rot, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(up_body, [0.0, 1.0, 0.0], atol=1e-12), up_body

    up_id, goal_id = pl.cone_frame_anchors(Rotation.identity(),
                                           np.array([2.0, 0.0, 0.0]))
    assert np.allclose(up_id, [0.0, 0.0, 1.0], atol=1e-12), up_id
    # Goal is the world origin, so from +x the goal direction is -x.
    assert np.allclose(goal_id, [-1.0, 0.0, 0.0], atol=1e-12), goal_id

    # On the goal the direction is undefined and falls back to +x.
    _, goal_at_goal = pl.cone_frame_anchors(Rotation.identity(), np.zeros(3))
    assert np.allclose(goal_at_goal, [1.0, 0.0, 0.0], atol=1e-12), goal_at_goal


def test_frame_anchoring():
    up, fb = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    t1, t2 = pl.tangent_basis_anchored(np.array([1.0, 0.0, 0.0]), up, fb)
    assert np.allclose(t1, [0.0, 1.0, 0.0], atol=1e-12), t1
    assert np.allclose(t2, [0.0, 0.0, 1.0], atol=1e-12), t2

    rng = np.random.RandomState(0)
    for n in rand_unit(rng, 300):
        t1, t2 = pl.tangent_basis_anchored(n, up, fb)
        for a, b in ((t1, t2), (t1, n), (t2, n)):
            assert abs(float(np.dot(a, b))) < 1e-9
        assert abs(np.linalg.norm(t1) - 1.0) < 1e-9
        assert abs(np.linalg.norm(t2) - 1.0) < 1e-9
        # {t1, t2, n} right-handed.
        assert np.allclose(np.cross(t1, t2), n, atol=1e-9)

        # t2 is the anchor PROJECTED into the tangent plane, not the anchor. That
        # is why it equals world up only when n is perpendicular to up.
        c = abs(float(np.dot(n, up)))
        w = float(np.clip((c - pl.CONE_FRAME_BLEND_COS_LO)
                          / (pl.CONE_FRAME_BLEND_COS_HI - pl.CONE_FRAME_BLEND_COS_LO),
                          0.0, 1.0))
        anchor = (1.0 - w) * up + w * fb
        anchor = anchor / np.linalg.norm(anchor)
        proj = anchor - float(np.dot(n, anchor)) * n
        assert np.allclose(t2, proj / np.linalg.norm(proj), atol=1e-9)


def test_blend_band():
    up, fb = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])

    # Outside the band the anchor is pure world up; past cos_hi it is pure goal.
    n_lo = np.array([math.sqrt(1 - 0.5 ** 2), 0.0, 0.5])          # |n.up| = 0.50
    t1_lo, _ = pl.tangent_basis_anchored(n_lo, up, fb)
    assert np.allclose(t1_lo, np.cross(up, n_lo) / np.linalg.norm(np.cross(up, n_lo)),
                       atol=1e-12)
    n_hi = np.array([math.sqrt(1 - 0.99 ** 2), 0.0, 0.99])        # |n.up| = 0.99
    t1_hi, _ = pl.tangent_basis_anchored(n_hi, up, fb)
    assert np.allclose(t1_hi, np.cross(fb, n_hi) / np.linalg.norm(np.cross(fb, n_hi)),
                       atol=1e-12)

    # Sweeping through the band the frame must not flip or jump. The fallback
    # here is perpendicular to the plane the normal sweeps in; the coplanar case
    # is a genuine singularity and is pinned by test_coplanar_flip.
    fb_generic = np.array([0.0, 1.0, 0.0])
    prev = None
    for cz in np.linspace(0.60, 0.999, 500):
        n = np.array([math.sqrt(max(1.0 - cz * cz, 0.0)), 0.0, cz])
        n = n / np.linalg.norm(n)
        t1, t2 = pl.tangent_basis_anchored(n, up, fb_generic)
        if prev is not None:
            assert float(np.dot(t1, prev[0])) > 0.99, cz
            assert float(np.dot(t2, prev[1])) > 0.99, cz
        prev = (t1, t2)


def test_coplanar_flip():
    """Pin a real discontinuity in the sim's anchored basis, so neither side
    'fixes' it without the other noticing.

    tangent_basis_anchored builds t1 = anchor x n. The anchor blends from up to
    the (horizontal) goal direction, so it sweeps the vertical plane containing
    the goal. When n lies in that same plane, the blend drives the anchor THROUGH
    n, t1 passes through zero and flips sign, and the lateral push direction
    reverses. Everything stays unit-norm, so no other assert in this file notices.

    Reachable, not merely theoretical: a flat face gives all 128 points one
    normal, and the face the oracle picks is the one facing away from the goal,
    i.e. already near that plane. It needs the face tilted about 35 deg from
    vertical to coincide. The trained policy saw the same flip, so deployment
    matches the sim; the risk is that near this configuration the SIGN of the
    lateral push is hypersensitive to pose error.
    """
    up, fb = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])

    def t1_at(cz):
        n = np.array([math.sqrt(1.0 - cz * cz), 0.0, cz])
        return pl.tangent_basis_anchored(n, up, fb)[0]

    before, after = t1_at(0.8155), t1_at(0.8175)
    assert np.allclose(before, [0.0, 1.0, 0.0], atol=1e-9), before
    assert np.allclose(after, [0.0, -1.0, 0.0], atol=1e-9), after

    sim_tba = _load_sim_tba()
    if sim_tba is None:
        return False
    import torch
    for cz in (0.8155, 0.8165, 0.8175):
        n = np.array([[math.sqrt(1.0 - cz * cz), 0.0, cz]])
        t1_s, _ = sim_tba(torch.as_tensor(n), torch.as_tensor(up[None, :]),
                          torch.as_tensor(fb[None, :]),
                          cos_lo=pl.CONE_FRAME_BLEND_COS_LO,
                          cos_hi=pl.CONE_FRAME_BLEND_COS_HI)
        assert np.allclose(t1_at(cz), t1_s[0].numpy(), atol=1e-9), cz
    return True


def test_cone_bound():
    rng = np.random.RandomState(1)
    normals = rand_unit(rng, 500)
    up, fb = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    max_ang = 0.0
    for n in normals:
        a_m = rng.normal() * 4.0
        a_tan = rng.normal(size=2) * 3.0
        f, decoded = pl.decode_cone_action(a_m, a_tan, n, up, fb)

        mag = float(np.linalg.norm(f))
        assert pl.FORCE_MIN - 1e-9 <= mag <= pl.FORCE_MAX + 1e-9, mag
        # Strictly into the surface: the cone is a closed subset of the open
        # hemisphere, so unlike the old projection this can never come out zero.
        assert float(np.dot(f, n)) < 0.0

        ang = math.degrees(math.acos(float(np.clip(np.dot(f / mag, -n), -1.0, 1.0))))
        assert ang <= pl.CONE_THETA_MAX_DEG + 1e-9, ang
        max_ang = max(max_ang, ang)

        assert abs(decoded[0] - mag / pl.FORCE_MAX) < 1e-12
        assert float(np.linalg.norm(decoded[1:3])) < 1.0
    return max_ang


def test_encode_round_trip():
    rng = np.random.RandomState(2)
    up, fb = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    n_clamped = 0
    for n in rand_unit(rng, 400):
        t1, t2 = pl.tangent_basis_anchored(n, up, fb)
        mag = float(rng.uniform(pl.FORCE_MIN + 1e-3, pl.FORCE_MAX - 1e-3))

        # A direction built inside the cone must round trip exactly.
        u = rng.normal(size=2)
        u = u / np.linalg.norm(u) * rng.uniform(0.0, 0.9)
        d_in = -n + pl.CONE_TAN_THETA_MAX * (u[0] * t1 + u[1] * t2)
        d_in = d_in / np.linalg.norm(d_in)
        a_m, a_tan = orc.cone_encode(d_in, mag, n, up, fb)
        f, _ = pl.decode_cone_action(a_m, a_tan, n, up, fb)
        assert np.allclose(f / np.linalg.norm(f), d_in, atol=1e-9)
        assert abs(float(np.linalg.norm(f)) - mag) < 1e-9

        # An out-of-cone direction must land ON the boundary, not be rejected.
        d_out = rand_unit(rng, 1)[0]
        if float(np.dot(d_out, -n)) <= 1e-6:
            continue
        ang_req = math.degrees(math.acos(float(np.clip(np.dot(d_out, -n), -1.0, 1.0))))
        if ang_req <= pl.CONE_THETA_MAX_DEG:
            continue
        n_clamped += 1
        a_m, a_tan = orc.cone_encode(d_out, mag, n, up, fb)
        f, _ = pl.decode_cone_action(a_m, a_tan, n, up, fb)
        ang = math.degrees(math.acos(
            float(np.clip(np.dot(f / np.linalg.norm(f), -n), -1.0, 1.0))))
        cap = math.degrees(math.atan(orc.U_MAX * pl.CONE_TAN_THETA_MAX))
        assert abs(ang - cap) < 1e-6, (ang, cap)

    # A pull request must degrade to a straight inward push, not blow up.
    n = np.array([0.0, 0.0, 1.0])
    a_m, a_tan = orc.cone_encode(n, 40.0, n, up, fb)
    f, _ = pl.decode_cone_action(a_m, a_tan, n, up, fb)
    assert np.allclose(f / np.linalg.norm(f), -n, atol=1e-12), f
    return n_clamped


def _load_sim_tba():
    """The sim's tangent_basis_anchored, pulled out of obs_noise.py by source.

    obs_noise.py imports isaaclab at module level, so it is not importable here.
    Compiling just this one function keeps the cross-check honest: it runs the
    sim's actual source text rather than a transcription of it.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    path = os.path.join(SIM_REPO, "obs_noise.py")
    if not os.path.isfile(path):
        return None
    import __future__
    import ast
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    ns = {"torch": __import__("torch"), "math": math}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "tangent_basis_anchored":
            mod = ast.Module(body=[node], type_ignores=[])
            # The file relies on `from __future__ import annotations` for its
            # tuple[...] return annotation, which python 3.8 cannot evaluate.
            exec(compile(mod, path, "exec", __future__.annotations.compiler_flag), ns)
            return ns["tangent_basis_anchored"]
    return None


def test_against_sim():
    """Compare against the sim's own torch code, which is the real proof."""
    sim_tba = _load_sim_tba()
    if sim_tba is None:
        print("  SKIP sim cross-check: torch or %s unavailable" % SIM_REPO)
        return False
    import torch

    rng = np.random.RandomState(3)
    N = 256
    n = rand_unit(rng, N)
    up = rand_unit(rng, N)
    fb = rand_unit(rng, N)
    a_m = rng.normal(size=N) * 3.0
    a_tan = rng.normal(size=(N, 2)) * 2.0

    t1_s, t2_s = sim_tba(torch.as_tensor(n), torch.as_tensor(up), torch.as_tensor(fb),
                         cos_lo=pl.CONE_FRAME_BLEND_COS_LO,
                         cos_hi=pl.CONE_FRAME_BLEND_COS_HI)
    t1_s, t2_s = t1_s.numpy(), t2_s.numpy()

    # Transcription of objectmanip_env_discrete._decode_cone_action.
    u_s = a_tan / np.sqrt(1.0 + (a_tan * a_tan).sum(axis=-1, keepdims=True))
    d_s = -n + pl.CONE_TAN_THETA_MAX * (u_s[:, 0:1] * t1_s + u_s[:, 1:2] * t2_s)
    d_s = d_s / np.linalg.norm(d_s, axis=-1, keepdims=True)
    mag_s = pl.FORCE_MIN + (pl.FORCE_MAX - pl.FORCE_MIN) / (1.0 + np.exp(-a_m))
    f_s = mag_s[:, None] * d_s

    for i in range(N):
        t1, t2 = pl.tangent_basis_anchored(n[i], up[i], fb[i])
        assert np.allclose(t1, t1_s[i], atol=1e-6), (i, t1, t1_s[i])
        assert np.allclose(t2, t2_s[i], atol=1e-6), (i, t2, t2_s[i])
        f, _ = pl.decode_cone_action(a_m[i], a_tan[i], n[i], up[i], fb[i])
        assert np.allclose(f, f_s[i], atol=1e-6), (i, f, f_s[i])
    return True


def main():
    test_row_vs_column()
    test_frame_anchoring()
    test_blend_band()
    flip_checked = test_coplanar_flip()
    max_ang = test_cone_bound()
    n_clamped = test_encode_round_trip()
    crossed = test_against_sim()
    print("cone_lib OK: theta_max=%.1f deg (worst sampled %.2f), "
          "|F| in [%.0f, %.0f] N, %d out-of-cone encodes clamped, sim cross-check %s"
          % (pl.CONE_THETA_MAX_DEG, max_ang, pl.FORCE_MIN, pl.FORCE_MAX,
             n_clamped, "PASSED" if crossed else "skipped"))
    print("  coplanar t1 sign flip at |n.up|~0.8165 reproduced%s"
          % (" and matched against the sim" if flip_checked else ""))


if __name__ == "__main__":
    main()

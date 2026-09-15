#!/usr/bin/env python3
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from npm_policy import policy_lib as pl

TAPER_DIST = 1.0
FORCE_MIN_TAPER = 25.0
P_FAR = 0.1
P_NEAR = 0.8
DISTS = [0.0, 0.05, 0.2, 0.5, 0.9, 1.0, 2.0]


def test_gain_is_monotonic():
    gains = [pl.taper_gain(d, TAPER_DIST) for d in DISTS]
    assert all(0.0 <= g <= 1.0 for g in gains), gains
    assert all(a <= b for a, b in zip(gains, gains[1:])), gains
    assert pl.taper_gain(TAPER_DIST, TAPER_DIST) == 1.0
    assert pl.taper_gain(2.0 * TAPER_DIST, TAPER_DIST) == 1.0
    print("PASS: gain rises with the distance to the goal and saturates at 1.0")


def test_prob_rises_towards_the_goal_and_never_reaches_one():
    probs = [pl.taper_prob(d, TAPER_DIST, P_FAR, P_NEAR) for d in DISTS]
    assert all(b <= a for a, b in zip(probs, probs[1:])), probs
    assert all(0.0 <= p < 1.0 for p in probs), probs
    assert abs(probs[0] - P_NEAR) < 1e-12, probs[0]
    assert abs(probs[-1] - P_FAR) < 1e-12, probs[-1]
    assert pl.taper_prob(0.0, TAPER_DIST, P_FAR, 1.0) == pl.TAPER_P_CAP
    print("PASS: taper probability is %.2f at the goal, %.2f at %.1f m, always < 1"
          % (probs[0], probs[-1], TAPER_DIST))


def test_magnitude_bounds_and_direction():
    f = np.array([0.0, 0.0, -pl.FORCE_MAX])
    for d in DISTS:
        for seed in range(32):
            rng = np.random.default_rng(seed)
            ft = pl.apply_force_taper(f, d, TAPER_DIST, FORCE_MIN_TAPER, P_FAR,
                                      P_NEAR, rng)
            mag = float(np.linalg.norm(ft))
            assert FORCE_MIN_TAPER - 1e-9 <= mag <= pl.FORCE_MAX + 1e-9, (d, mag)
            assert np.allclose(ft / mag, f / np.linalg.norm(f), atol=1e-12), ft
    print("PASS: tapered magnitude stays in [%.0f, %.0f] N and the direction is kept"
          % (FORCE_MIN_TAPER, pl.FORCE_MAX))


def test_taper_rate_rises_towards_the_goal():
    f = np.array([0.0, 0.0, -pl.FORCE_MAX])
    n = 4000
    rates = []
    for d in [2.0, 1.0, 0.5, 0.0]:
        rng = np.random.default_rng(7)
        hits = sum(
            float(np.linalg.norm(
                pl.apply_force_taper(f, d, TAPER_DIST, FORCE_MIN_TAPER, P_FAR,
                                     P_NEAR, rng))) < pl.FORCE_MAX - 1e-9
            for _ in range(n))
        rates.append(hits / float(n))
    assert all(a <= b + 0.02 for a, b in zip(rates, rates[1:])), rates
    assert rates[0] < 0.2 and rates[-1] > 0.7, rates
    print("PASS: taper rate %s over 2.0/1.0/0.5/0.0 m"
          % [round(r, 3) for r in rates])


def test_seeded_stream_is_reproducible():
    f = np.array([1.0, -2.0, -3.0])
    f = pl.FORCE_MAX * f / np.linalg.norm(f)
    a = [pl.apply_force_taper(f, 0.3, TAPER_DIST, FORCE_MIN_TAPER, P_FAR, P_NEAR,
                              np.random.default_rng(3)) for _ in range(5)]
    b = [pl.apply_force_taper(f, 0.3, TAPER_DIST, FORCE_MIN_TAPER, P_FAR, P_NEAR,
                              np.random.default_rng(3)) for _ in range(5)]
    assert np.allclose(a, b, atol=1e-12)
    print("PASS: a seeded rng reproduces the same taper decisions")


def test_disabled_path_is_bit_identical():
    f = np.array([0.0, 0.0, -pl.FORCE_MAX])
    for d in DISTS:
        ft = pl.apply_force_taper(f, d, TAPER_DIST, FORCE_MIN_TAPER, 0.0, 0.0,
                                  np.random.default_rng(0))
        assert np.array_equal(ft, f), (d, ft)
    print("PASS: a zero probability leaves the policy force bit-identical")


def main():
    test_gain_is_monotonic()
    test_prob_rises_towards_the_goal_and_never_reaches_one()
    test_magnitude_bounds_and_direction()
    test_taper_rate_rises_towards_the_goal()
    test_seeded_stream_is_reproducible()
    test_disabled_path_is_bit_identical()


if __name__ == "__main__":
    main()

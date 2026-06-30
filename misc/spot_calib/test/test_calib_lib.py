#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from spot_calib import calib_lib as cl


def rand_R(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return cl.quat_wxyz_to_rotmat(q)


def rand_se3(rng, t=0.3):
    return cl.make_se3(rand_R(rng), rng.uniform(-t, t, 3))


def rot_angle(R):
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))


def main():
    rng = np.random.default_rng(0)
    X = rand_se3(rng)
    A = [rand_se3(rng) for _ in range(10)]
    B = [cl.se3_inv(X) @ a @ X for a in A]

    Xh = cl.solve_axxb(A, B)
    ang = rot_angle(X[:3, :3].T @ Xh[:3, :3])
    terr = np.linalg.norm(X[:3, 3] - Xh[:3, 3])
    assert ang < 1e-6, ang
    assert terr < 1e-6, terr

    Bn = []
    for b in B:
        dR = cl.quat_wxyz_to_rotmat(
            (lambda q: q / np.linalg.norm(q))(np.array([1, *(0.003 * rng.normal(size=3))])))
        nb = b.copy()
        nb[:3, :3] = dR @ b[:3, :3]
        nb[:3, 3] = b[:3, 3] + 0.003 * rng.normal(size=3)
        Bn.append(nb)
    Xn = cl.solve_axxb(A, Bn)
    ang_n = rot_angle(X[:3, :3].T @ Xn[:3, :3])
    terr_n = np.linalg.norm(X[:3, 3] - Xn[:3, 3])
    assert ang_n < 0.05 and terr_n < 0.05, (ang_n, terr_n)

    print("PASS: exact recovery ang=%.2e t=%.2e ; noisy ang=%.4f t=%.4f" % (
        ang, terr, ang_n, terr_n))


if __name__ == "__main__":
    main()

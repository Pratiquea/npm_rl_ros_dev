#!/usr/bin/env python3
"""
Offline tests for the object-state Kalman filters (no ROS needed).

Example:
    /usr/bin/python3 test_kalman.py
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from npm_control.kalman import ConstantVelocityKF, AngularRateEstimator

DT = 1.0 / 120.0            # mocap sample period
SETTLE_VEL = 0.1            # raw/smdp_contract.md settle_vel_threshold
# Defaults from npm.yaml. Upstream's sigma_a=20 fails test_linear_settles_to_zero.
SIGMA_A, SIGMA_MEAS = 1.0, 0.001


def test_linear_converges():
    rng = np.random.default_rng(0)
    truth_v = np.array([0.35, -0.20, 0.05])
    kf = ConstantVelocityKF(SIGMA_A, SIGMA_MEAS)
    p = np.array([1.0, 0.5, 0.2])
    kf.reset(p)
    vels = []
    for i in range(360):                       # 3 s of samples
        p = p + truth_v * DT
        kf.update(p + rng.normal(scale=SIGMA_MEAS, size=3), DT)
        if i > 240:
            vels.append(kf.vel().copy())
    # Averaged over the tail: a single estimate still carries a few mm/s of
    # jitter, which is by design (see the sigma_a note in kalman.py).
    err = np.linalg.norm(np.mean(vels, axis=0) - truth_v) / np.linalg.norm(truth_v)
    assert err < 0.05, (np.mean(vels, axis=0), err)
    assert np.linalg.norm(kf.pos() - p) < 5e-3, (kf.pos(), p)
    print("PASS linear velocity converges (rel err %.4f)" % err)


def test_linear_settles_to_zero():
    # A stationary object must sit well under the settle threshold: this is what
    # optitrack_bridge's sigma_a=20 default fails, at ~0.38 m/s.
    rng = np.random.default_rng(1)
    kf = ConstantVelocityKF(SIGMA_A, SIGMA_MEAS)
    p = np.array([0.3, 0.3, 0.1])
    kf.reset(p)
    worst = 0.0
    for i in range(600):
        kf.update(p + rng.normal(scale=SIGMA_MEAS, size=3), DT)
        if i > 120:
            worst = max(worst, float(np.linalg.norm(kf.vel())))
    assert worst < 0.5 * SETTLE_VEL, worst
    print("PASS stationary object reads %.4f m/s, under the %.2f threshold"
          % (worst, SETTLE_VEL))


def test_angular_converges():
    omega = np.array([0.0, 0.0, 1.2])          # rad/s yaw spin
    est = AngularRateEstimator(SIGMA_A, SIGMA_MEAS)
    rot = Rotation.identity()
    est.reset(rot)
    out = np.zeros(3)
    for _ in range(240):
        rot = Rotation.from_rotvec(omega * DT) * rot
        out = est.update(rot, DT)
    err = np.linalg.norm(out - omega) / np.linalg.norm(omega)
    assert err < 0.05, (out, err)
    print("PASS angular velocity converges (rel err %.4f)" % err)


def test_angular_axis():
    # A tilted spin axis must come back in world frame, not body frame.
    omega = np.array([0.4, -0.3, 0.9])
    est = AngularRateEstimator(SIGMA_A, SIGMA_MEAS)
    rot = Rotation.from_euler("y", 40.0, degrees=True)
    est.reset(rot)
    out = np.zeros(3)
    for _ in range(360):
        rot = Rotation.from_rotvec(omega * DT) * rot     # pre-multiply: world frame
        out = est.update(rot, DT)
    assert np.linalg.norm(out - omega) / np.linalg.norm(omega) < 0.05, out
    print("PASS angular velocity axis is world-frame")


def test_reset_has_no_spike():
    # After a dropout the caller resets; the next sample must not emit a spike.
    kf = ConstantVelocityKF(SIGMA_A, SIGMA_MEAS)
    kf.reset(np.zeros(3))
    for i in range(120):
        kf.update(np.array([0.5, 0.0, 0.0]) * (i + 1) * DT, DT)
    kf.reset(np.array([5.0, 0.0, 0.0]))        # object teleports after the gap
    assert np.linalg.norm(kf.vel()) == 0.0, kf.vel()
    _p, v = kf.update(np.array([5.0, 0.0, 0.0]), DT)
    assert np.linalg.norm(v) < 0.1, v
    print("PASS reset after a dropout emits no velocity spike")


def main():
    test_linear_converges()
    test_linear_settles_to_zero()
    test_angular_converges()
    test_angular_axis()
    test_reset_has_no_spike()
    print("ALL PASS")


if __name__ == "__main__":
    main()

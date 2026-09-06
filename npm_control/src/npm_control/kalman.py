#!/usr/bin/env python3
"""
Constant-velocity Kalman filtering for the mocap object state. Pure
numpy/scipy, no ROS, so it is unit-testable offline.
"""
import numpy as np


class ConstantVelocityKF(object):
    """6-state [p(3), v(3)] Kalman filter with a white-acceleration model.

    Ported from optitrack_bridge's LinearKalmanFilter, with the noise terms
    exposed instead of hardcoded. Used on position directly, and on an
    accumulated rotation vector by AngularRateEstimator.

    sigma_a is the dominant knob. Upstream's 20.0 makes the filter trust 1 mm
    mocap noise so hard that a STATIONARY object reads ~0.38 m/s, which never
    clears the 0.1 m/s settle threshold; 1.0 gives ~0.035 m/s at rest and still
    locks onto a 3 m/s^2 push inside a second. Raise sigma_meas (not sigma_a) if
    the real marker noise turns out worse than 1 mm.
    """

    def __init__(self, sigma_a, sigma_meas):
        self.sigma_a = float(sigma_a)
        self.sigma_meas = float(sigma_meas)
        self.x = None
        self.P = None

    def reset(self, z):
        self.x = np.concatenate([np.asarray(z, dtype=np.float64).reshape(3),
                                 np.zeros(3)])
        # Position is trusted (it is the measurement); velocity is unknown.
        self.P = np.diag([self.sigma_meas ** 2] * 3 + [1.0] * 3)

    def update(self, z, dt):
        z = np.asarray(z, dtype=np.float64).reshape(3)
        if self.x is None or dt <= 0.0:
            self.reset(z)
            return self.pos(), self.vel()

        I3 = np.eye(3)
        F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
        Q = (self.sigma_a ** 2) * np.block(
            [[dt ** 4 / 4.0 * I3, dt ** 3 / 2.0 * I3],
             [dt ** 3 / 2.0 * I3, dt ** 2 * I3]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        H = np.hstack([I3, np.zeros((3, 3))])
        R = (self.sigma_meas ** 2) * I3
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.pos(), self.vel()

    def pos(self):
        return np.zeros(3) if self.x is None else self.x[:3].copy()

    def vel(self):
        return np.zeros(3) if self.x is None else self.x[3:].copy()


class AngularRateEstimator(object):
    """World-frame angular velocity from a stream of orientations.

    Per-sample shortest-arc increments are accumulated into an unwrapped
    rotation vector, which the constant-velocity filter then treats as a
    position; its velocity state is the angular velocity. Valid because
    increments at mocap rate are small, so treating them as additive (rotations
    do not commute) costs O(dt^2).
    """

    def __init__(self, sigma_alpha, sigma_meas):
        self.kf = ConstantVelocityKF(sigma_alpha, sigma_meas)
        self.phi = np.zeros(3)
        self.prev_rot = None

    def reset(self, rot=None):
        self.phi = np.zeros(3)
        self.prev_rot = rot
        self.kf.reset(self.phi)

    def update(self, rot, dt):
        if self.prev_rot is None or dt <= 0.0:
            self.reset(rot)
            return np.zeros(3)
        self.phi = self.phi + (rot * self.prev_rot.inv()).as_rotvec()
        self.prev_rot = rot
        _phi, omega = self.kf.update(self.phi, dt)
        return omega

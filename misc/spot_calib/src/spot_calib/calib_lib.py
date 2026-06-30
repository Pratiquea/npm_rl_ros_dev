#!/usr/bin/env python3
"""

Model:  A_i = Y . B_i . X
  A_i = T_{world<-spot_body_mocap}(i)   (mocap stream)
  B_i = T_{vision <-body}(i)              (spot SDK stream)
  Xb  = T_{spot_body_mocap<-spot_body_link_internal}   (constant, "body extrinsic")
  Y   = T_{world<-vision}                                (constant, "world link")
  with X = inv(Xb), so A_i = Y . B_i . inv(Xb).

Compute both Xb and Y from cv2.calibrateRobotWorldHandEye call.
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import correlate, correlation_lags

def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    # Note: scipy normalizes
    return Rotation.from_quat([x, y, z, w]).as_matrix()

def rotmat_to_quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])

def make_se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T

def se3_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def rotmat_to_rotvec(R):
    return Rotation.from_matrix(R).as_rotvec()

def relative_motions_body(poses):
    """
    Right relatives  inv(pose_i) @ pose_{i+1}  -> body-frame relative motion.
    NOTE to myself: the earlier version used left relatives pose_{i+1}@inv(pose_i), which solve for Y, not Xb.
    """
    return [se3_inv(poses[i]) @ poses[i + 1] for i in range(len(poses) - 1)]

def relative_motions_world(poses):
    """
    Left relatives  pose_{i+1} @ inv(pose_i)  -> world-frame relative motion.
    """
    return [poses[i + 1] @ se3_inv(poses[i]) for i in range(len(poses) - 1)]


# ### solvers
def solve_robot_world(A_abs, B_abs, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH):
    """
    A_abs: T_{world<-mocap_body}, B_abs: T_{vision<-body} (absolute poses).
    Returns (Xb, Y):
      Xb = T_{spot_body_mocap<-spot_body_link_internal}
      Y  = T_{world<-vision}
    """
    assert len(A_abs) == len(B_abs) >= 3, "need >=3 paired poses"
    Rb, tb, Rg, tg = cv2.calibrateRobotWorldHandEye(
        [p[:3, :3] for p in A_abs], [p[:3, 3] for p in A_abs],
        [p[:3, :3] for p in B_abs], [p[:3, 3] for p in B_abs], method=method)
    Xb = make_se3(Rb, np.asarray(tb).reshape(3))
    Y  = make_se3(Rg, np.asarray(tg).reshape(3))
    return Xb, Y

def solve_axxb_manual(A_abs, B_abs):
    """
    Cross-check: AX=XB on RIGHT relatives -> Xb (Park-style rotvec
    Procrustes + linear translation).
    """
    A = relative_motions_body(A_abs)
    B = relative_motions_body(B_abs)
    H = np.zeros((3, 3))
    for a, b in zip(A, B):
        H += np.outer(rotmat_to_rotvec(a[:3, :3]), rotmat_to_rotvec(b[:3, :3]))
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rx = U @ D @ Vt
    C, d = [], []
    for a, b in zip(A, B):
        C.append(a[:3, :3] - np.eye(3))
        d.append(Rx @ b[:3, 3] - a[:3, 3])
    tx, *_ = np.linalg.lstsq(np.vstack(C), np.concatenate(d), rcond=None)
    return make_se3(Rx, tx)

def residuals(A_abs, B_abs, Xb, Y):
    """
    Reconstruct A_i = Y . B_i . inv(Xb); return mean rot(deg) & trans(m) error.
    """
    rot, trans = [], []
    Xinv = se3_inv(Xb)
    for a, b in zip(A_abs, B_abs):
        e = se3_inv(a) @ (Y @ b @ Xinv)
        rot.append(np.degrees(np.linalg.norm(rotmat_to_rotvec(e[:3, :3]))))
        trans.append(np.linalg.norm(e[:3, 3]))
    return float(np.mean(rot)), float(np.mean(trans))

# temporal alignment
def interp_se3(T0, T1, u):
    """
    Interpolate two SE3 poses: SLERP rotation, lerp translation. u in [0,1].
    """
    key = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    R = Slerp([0.0, 1.0], key)(float(u))
    t = (1.0 - u) * T0[:3, 3] + u * T1[:3, 3]
    return make_se3(R.as_matrix(), t)

def resample_stream(stamps, poses, query):
    """
    Interpolate an SE3 series (stamps[] seconds, poses[] 4x4) onto query times.
    Returns (q_t[], poses_out[]) for the subset of query within [stamps].
    """
    stamps = np.asarray(stamps, float)
    order = np.argsort(stamps)
    stamps = stamps[order]
    poses = [poses[i] for i in order]
    out_t, out_p = [], []
    for qt in query:
        if qt < stamps[0] or qt > stamps[-1]:
            # no extrapolation, so skip
            continue                       
        j = int(np.searchsorted(stamps, qt))
        if j == 0:
            out_t.append(qt)
            out_p.append(poses[0])
            continue
        t0, t1 = stamps[j - 1], stamps[j]
        u = 0.0 if t1 == t0 else (qt - t0) / (t1 - t0)
        out_t.append(qt)
        out_p.append(interp_se3(poses[j - 1], poses[j], u))
    return np.asarray(out_t), out_p

def angular_speed_series(stamps, poses):
    """
    |angular_vel| (rad/s) between consecutive poses 
    Returns (mid_stamps[], speeds[]).
    """
    stamps = np.asarray(stamps, float)
    mt, sp = [], []
    for i in range(len(poses) - 1):
        dt = stamps[i + 1] - stamps[i]
        if dt <= 0:
            continue
        dR = poses[i][:3, :3].T @ poses[i + 1][:3, :3]
        mt.append(0.5 * (stamps[i] + stamps[i + 1]))
        sp.append(np.linalg.norm(rotmat_to_rotvec(dR)) / dt)
    return np.asarray(mt), np.asarray(sp)

def estimate_td_xcorr(tA, A, tB, B, max_td=0.1, dt=0.005):
    """
    Cross-correlate angular-speed of the two streams on a uniform grid.
    Returns td (s): the matching B event is at tB ~ tA + td (B stamped later
    than A by td when positive).
    """
    mtA, spA = angular_speed_series(tA, A)
    mtB, spB = angular_speed_series(tB, B)
    if len(spA) < 3 or len(spB) < 3:
        return 0.0
    t0 = max(mtA[0], mtB[0])
    t1 = min(mtA[-1], mtB[-1])
    if t1 <= t0:
        return 0.0
    grid = np.arange(t0, t1, dt)
    a = np.interp(grid, mtA, spA)
    b = np.interp(grid, mtB, spB)
    # Sliding-window normalized cross-correlation.
    # Raw correlate biases toward lag 0 for the smooth, rectified |omega| signal;
    # per-lag mean/variance removal makes it unbiased.
    oa, ob = np.ones_like(a), np.ones_like(b)
    s_ab = correlate(a, b, "full")
    s_a = correlate(a, ob, "full"); s_b = correlate(oa, b, "full")
    s_aa = correlate(a * a, ob, "full"); s_bb = correlate(oa, b * b, "full")
    cnt = np.maximum(correlate(oa, ob, "full"), 1.0)
    lags = correlation_lags(len(a), len(b), "full")
    cov = s_ab - s_a * s_b / cnt
    va = np.maximum(s_aa - s_a ** 2 / cnt, 1e-12)
    vb = np.maximum(s_bb - s_b ** 2 / cnt, 1e-12)
    ncc = cov / np.sqrt(va * vb)
    K = int(max_td / dt)
    mask = (np.abs(lags) <= K) & (cnt >= 3)
    L = int(lags[mask][np.argmax(ncc[mask])])
    return -L * dt

def estimate_td_grid(A, tA, tB, B, td_range=(-0.1, 0.1), step=0.005):
    """Sweep td, resample B onto (tA+td), solve robot-world, pick td with min
    residual. Returns (best_td, best_rot_deg, best_trans_m)."""
    best = (0.0, np.inf, np.inf)
    td = td_range[0]
    while td <= td_range[1] + 1e-9:
        qt, Bq = resample_stream(tB, B, np.asarray(tA, float) + td)
        if len(Bq) >= 3:
            keep = {round(float(t), 9): k for k, t in enumerate(qt)}
            Apair, Bpair = [], []
            for k, t in enumerate(tA):
                j = keep.get(round(float(t) + td, 9))
                if j is not None:
                    Apair.append(A[k])
                    Bpair.append(Bq[j])
            if len(Apair) >= 3:
                Xb, Y = solve_robot_world(Apair, Bpair)
                r, tr = residuals(Apair, Bpair, Xb, Y)
                if r + np.degrees(tr) < best[1] + np.degrees(best[2]):
                    best = (float(td), r, tr)
        td += step
    return best

# degeneracy / excitation meter
def excitation(A_abs, min_rot_deg=5.0, n_target=12, l2_min=0.15):
    """
    Quantify if the collected motion escapes the hand-eye degeneracy
    (needs >=2 non-parallel rotation axes).
    """
    axes, w = [], []
    for M in relative_motions_world(A_abs):
        r = rotmat_to_rotvec(M[:3, :3])
        a = np.linalg.norm(r)
        if np.degrees(a) < min_rot_deg:
            continue
        axes.append(r / a)
        w.append(a)
    n = len(axes)
    sample_frac = min(1.0, n / float(n_target))
    if n < 2:
        return dict(n=n, l2r=0.0, l3r=0.0, spread_deg=0.0,
                    sample_frac=sample_frac, ready=False)
    A = np.array(axes)
    w = np.array(w)
    ev = np.sort(np.clip(np.linalg.eigvalsh((A * w[:, None]).T @ A), 0, None))[::-1]
    l2r = ev[1] / ev[0] if ev[0] > 0 else 0.0
    l3r = ev[2] / ev[0] if ev[0] > 0 else 0.0
    G = np.clip(np.abs(A @ A.T), -1, 1)
    spread = float(np.degrees(np.arccos(G.min())))
    ready = (n >= n_target) and (l2r >= l2_min)
    return dict(n=n, l2r=round(float(l2r), 3), l3r=round(float(l3r), 3),
                spread_deg=round(spread, 1), sample_frac=round(sample_frac, 2),
                ready=bool(ready))
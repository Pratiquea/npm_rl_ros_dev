#!/usr/bin/env python3
"""
Offline invariant tests for obs_lib (no ROS, no mocap needed).

Example:
    /usr/bin/python3 test_obs_lib.py [npz_path]
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from npm_policy import obs_lib as ol

NPZ = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/rwl-4090/gits/nonprehensile_object_manipulation/"
    "dataset/primitives/Paralelopiped/Paralelopiped.npz")


def _legacy_rotmat(q_wxyz):
    # The hand-rolled formula obs_math/policy_lib used before scipy. Kept here
    # only as the reference the refactor is checked against.
    w, x, y, z = q_wxyz
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2 * (yy + zz); R[0, 1] = 2 * (xy - wz);     R[0, 2] = 2 * (xz + wy)
    R[1, 0] = 2 * (xy + wz);     R[1, 1] = 1 - 2 * (xx + zz); R[1, 2] = 2 * (yz - wx)
    R[2, 0] = 2 * (xz - wy);     R[2, 1] = 2 * (yz + wx);     R[2, 2] = 1 - 2 * (xx + yy)
    return R


def test_identity_and_extras(pts, nrm, scale):
    pos = np.array([1.0, 0.0, 0.1])
    rot = Rotation.identity()
    obs = ol.assemble_observation(rot, pos, np.zeros(3), np.zeros(3), np.zeros(4),
                                  mass=20.0, friction=0.95, object_scale=scale,
                                  pts_norm=pts, normals=nrm)

    assert obs.shape[0] == ol.OBS_LEN, obs.shape
    assert obs.dtype == np.float32
    pc = obs[ol.PC_OFF:ol.NORMALS_OFF]
    normals_block = obs[ol.NORMALS_OFF:ol.EXTRAS_OFF]
    extras = obs[ol.EXTRAS_OFF:]
    assert np.allclose(pc, pts.reshape(-1), atol=1e-5), "identity rot must give pts_norm"
    assert np.allclose(normals_block, nrm.reshape(-1), atol=1e-5)
    assert extras.shape[0] == ol.EXTRAS_LEN

    object_pose = extras[ol.E_OBJECT_POSE:ol.E_OBJECT_POSE + 7]
    link_pose = extras[ol.E_LINK_POSE:ol.E_LINK_POSE + 7]
    assert np.allclose(extras[ol.E_PREV_ACTION:ol.E_PREV_ACTION + 4], 0.0)
    assert abs(extras[ol.E_MASS] - 20.0) < 1e-5
    assert np.allclose(object_pose[:3], pos, atol=1e-5), object_pose
    assert np.allclose(object_pose[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-5), "wxyz order"
    assert np.allclose(link_pose, object_pose, atol=1e-5), "v1 link == object"
    assert abs(extras[ol.E_FRICTION] - 0.95) < 1e-5
    assert abs(extras[ol.E_OBJECT_SCALE] - scale) < 1e-4
    assert abs(extras[ol.E_DIST_TO_GOAL] - 1.0) < 1e-5, extras[ol.E_DIST_TO_GOAL]
    assert np.allclose(extras[ol.E_DIR_TO_GOAL:ol.E_DIR_TO_GOAL + 2], [-1.0, 0.0],
                       atol=1e-5)
    assert np.all(np.isfinite(obs))
    print("PASS identity + extras layout")


def test_wxyz_slot_order(pts, nrm, scale):
    # 30 deg yaw: the pose slots must hold wxyz, not the scipy/ROS xyzw order.
    rot = Rotation.from_euler("z", 30.0, degrees=True)
    obs = ol.assemble_observation(rot, np.zeros(3), np.zeros(3), np.zeros(3),
                                  np.zeros(4), 20.0, 0.95, scale, pts, nrm)
    q = obs[ol.EXTRAS_OFF + ol.E_OBJECT_POSE + 3:ol.EXTRAS_OFF + ol.E_OBJECT_POSE + 7]
    assert abs(q[0] - np.cos(np.radians(15.0))) < 1e-6, q      # w first
    assert abs(q[3] - np.sin(np.radians(15.0))) < 1e-6, q      # z last
    assert abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6, q
    print("PASS wxyz slot order")


def test_velocity_and_goal_slots(pts, nrm, scale):
    pos = np.array([0.0, -2.0, 0.0])
    lin = np.array([0.1, -0.2, 0.3])
    ang = np.array([-0.4, 0.5, 0.6])
    obs = ol.assemble_observation(Rotation.identity(), pos, lin, ang,
                                  [1.0, 2.0, 3.0, 4.0], 20.0, 0.95, scale, pts, nrm)
    e = obs[ol.EXTRAS_OFF:]
    assert np.allclose(e[ol.E_LIN_VEL:ol.E_LIN_VEL + 3], lin, atol=1e-6)
    assert np.allclose(e[ol.E_ANG_VEL:ol.E_ANG_VEL + 3], ang, atol=1e-6)
    assert np.allclose(e[ol.E_PREV_ACTION:ol.E_PREV_ACTION + 4], [1, 2, 3, 4], atol=1e-6)
    assert abs(e[ol.E_DIST_TO_GOAL] - 2.0) < 1e-5
    # dir_to_goal points from the object back to the origin.
    assert np.allclose(e[ol.E_DIR_TO_GOAL:ol.E_DIR_TO_GOAL + 2], [0.0, 1.0], atol=1e-5)
    print("PASS velocity + goal slots")


def test_scipy_matches_legacy(pts, nrm):
    """The scipy rewrite must not move a single number the checkpoint sees."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q_wxyz = q if q[0] >= 0 else -q            # legacy formula sign convention
        rot = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

        R = _legacy_rotmat(q_wxyz)
        legacy_pc = (pts @ R.T).reshape(-1)
        legacy_nrm = (nrm @ R.T).reshape(-1)
        assert np.allclose(ol.rotate_cloud(rot, pts), legacy_pc, atol=1e-6)
        assert np.allclose(ol.rotate_cloud(rot, nrm), legacy_nrm, atol=1e-6)
        assert np.allclose(ol.quat_wxyz(rot), q_wxyz, atol=1e-12)
    print("PASS scipy path reproduces the legacy rotation")


def test_rotation_actually_rotates(pts, nrm, scale):
    rot = Rotation.from_euler("z", 90.0, degrees=True)
    obs = ol.assemble_observation(rot, np.zeros(3), np.zeros(3), np.zeros(3),
                                  np.zeros(4), 20.0, 0.95, scale, pts, nrm)
    pc = obs[ol.PC_OFF:ol.NORMALS_OFF].reshape(-1, 3)
    assert not np.allclose(pc, pts, atol=1e-3), "90 deg yaw must change the cloud"
    # +90 deg about z maps (x, y) -> (-y, x).
    assert np.allclose(pc[:, 0], -pts[:, 1], atol=1e-5)
    assert np.allclose(pc[:, 1], pts[:, 0], atol=1e-5)
    assert np.allclose(pc[:, 2], pts[:, 2], atol=1e-5)
    print("PASS 90 deg yaw rotates the cloud correctly")


def test_link_frame_cloud(pts, nrm, scale, centroid):
    """Must equal the sim's model_pcl_scaled_link_frame, and stay out of the obs."""
    link = ol.link_frame_cloud(pts, scale, centroid)
    sim_com_frame = pts * scale                              # env line 482
    sim_link_frame = sim_com_frame + centroid.reshape(1, 3)  # env lines 483-485
    assert link.shape == pts.shape, link.shape
    assert np.allclose(link, sim_link_frame, atol=1e-12)

    # The offset is the centroid itself, so an object whose COM sits off the link
    # origin gives a cloud the raw pts_norm cannot stand in for.
    assert np.allclose(link.mean(axis=0) - sim_com_frame.mean(axis=0), centroid,
                       atol=1e-9)

    # The observation keeps using unscaled, COM-centered points.
    obs = ol.assemble_observation(Rotation.identity(), np.zeros(3), np.zeros(3),
                                  np.zeros(3), np.zeros(4), 20.0, 0.95, scale,
                                  pts, nrm)
    assert np.allclose(obs[ol.PC_OFF:ol.NORMALS_OFF], pts.reshape(-1), atol=1e-5)
    print("PASS link_frame_cloud matches the sim, obs unchanged")


def test_shrink(pts, nrm, scale, centroid):
    """pcl_shrink contracts about the COM, and never leaks into the observation."""
    nominal = ol.link_frame_cloud(pts, scale, centroid)
    assert np.array_equal(ol.link_frame_cloud(pts, scale, centroid, 1.0), nominal), \
        "shrink=1.0 must be a bitwise passthrough"

    k = 0.97
    shrunk = ol.link_frame_cloud(pts, scale, centroid, k)

    # The whole premise of the parameterization: the scaling happens before the
    # centroid is added, so the contraction centre IS the COM and the extrinsic
    # needs no compensating offset. If this fails, the cloud has slid off the
    # object and every commanded contact point is wrong.
    assert np.allclose(shrunk.mean(axis=0), nominal.mean(axis=0), atol=1e-12)
    assert np.allclose(shrunk.mean(axis=0), centroid, atol=1e-6)

    # Every point moved toward the COM by exactly (1-k) of its offset.
    assert np.allclose(shrunk - centroid, k * (nominal - centroid), atol=1e-12)
    assert np.all(np.linalg.norm(shrunk - centroid, axis=1)
                  <= np.linalg.norm(nominal - centroid, axis=1) + 1e-12)

    # Reported margin is half the nominal extent times (1-k).
    inset = ol.shrink_inset(pts, scale, k)
    extent = (pts.max(axis=0) - pts.min(axis=0)) * scale
    assert np.allclose(inset, 0.5 * extent * (1.0 - k), atol=1e-12)
    assert np.allclose(ol.shrink_inset(pts, scale, 1.0), np.zeros(3), atol=1e-12)

    # The policy trained on the nominal scale, so the extras slot must not move.
    obs_nom = ol.assemble_observation(Rotation.identity(), np.zeros(3), np.zeros(3),
                                      np.zeros(3), np.zeros(4), 20.0, 0.95, scale,
                                      pts, nrm)
    assert abs(float(obs_nom[ol.EXTRAS_OFF + ol.E_OBJECT_SCALE]) - scale) < 1e-6
    print("PASS shrink contracts about the COM (inset %.1f/%.1f/%.1f mm at k=%.2f)"
          % (1000 * inset[0], 1000 * inset[1], 1000 * inset[2], k))


def test_cloud_pair(pts, scale, centroid):
    """The two published clouds: /npm/debug/obs_cloud and .../obs_cloud_shrunk.

    obs_cloud is nominal, obs_cloud_shrunk carries the margin. Their surfaces must
    differ by exactly the reported inset on every axis, per side.
    """
    k = 0.97
    nominal = ol.link_frame_cloud(pts, scale, centroid)
    shrunk = ol.link_frame_cloud(pts, scale, centroid, k)
    inset = ol.shrink_inset(pts, scale, k)

    # Each face moves inward by (1-k) of ITS OWN distance from the COM, because
    # the contraction centre is the COM. The two faces on one axis therefore move
    # by different amounts whenever the COM is off the bounding-box centre.
    hi = nominal.max(axis=0) - shrunk.max(axis=0)
    lo = shrunk.min(axis=0) - nominal.min(axis=0)
    assert np.allclose(hi, (1.0 - k) * (nominal.max(axis=0) - centroid), atol=1e-9)
    assert np.allclose(lo, (1.0 - k) * (centroid - nominal.min(axis=0)), atol=1e-9)

    # shrink_inset reports the mean of the two sides, which is what the startup
    # logs quote. Assert that relation so the logged number cannot drift from the
    # geometry it claims to describe.
    assert np.allclose(0.5 * (hi + lo), inset, atol=1e-9)
    ext_nom = nominal.max(axis=0) - nominal.min(axis=0)
    ext_shr = shrunk.max(axis=0) - shrunk.min(axis=0)
    assert np.allclose(ext_nom - ext_shr, 2.0 * inset, atol=1e-9)

    # The shrunk cloud is strictly inside the nominal one, which is the whole
    # point: no commanded point may sit on or past the real surface.
    assert np.all(shrunk.max(axis=0) < nominal.max(axis=0))
    assert np.all(shrunk.min(axis=0) > nominal.min(axis=0))
    print("PASS the cloud pair differs by %.1f/%.1f/%.1f mm per side (mean; the "
          "two faces differ by up to %.1f mm because the COM is off centre)"
          % (1000 * inset[0], 1000 * inset[1], 1000 * inset[2],
             1000 * float(np.abs(hi - lo).max())))


def test_model_info_effective_scale(scale):
    """The payload must catch a pcl_shrink disagreement, not just a wrong file."""
    payload = ol.format_model_info(NPZ, scale, 0.97)
    assert abs(float(payload.split()[3]) - scale * 0.97) < 1e-6

    ok, msg = ol.check_model_info(payload, NPZ, 0.97)
    assert ok, msg
    # Same npz, same sha1, different margin: the old sha1-only check passed this.
    ok, msg = ol.check_model_info(payload, NPZ, 1.0)
    assert not ok and "PCL_SHRINK MISMATCH" in msg, msg
    # A payload from before this field existed is reported as unchecked, not failed.
    ok, msg = ol.check_model_info(" ".join(payload.split()[:3]), NPZ, 0.97)
    assert ok and "UNCHECKED" in msg, msg
    print("PASS /npm/model_info carries the effective scale and catches a mismatch")


def main():
    pts, nrm, scale, centroid = ol.load_model_npz(NPZ)
    print("npz: pts%s normals%s scale=%.4f centroid=%s" %
          (pts.shape, nrm.shape, scale, np.round(centroid, 4)))
    test_identity_and_extras(pts, nrm, scale)
    test_wxyz_slot_order(pts, nrm, scale)
    test_velocity_and_goal_slots(pts, nrm, scale)
    test_scipy_matches_legacy(pts, nrm)
    test_rotation_actually_rotates(pts, nrm, scale)
    test_link_frame_cloud(pts, nrm, scale, centroid)
    test_shrink(pts, nrm, scale, centroid)
    test_cloud_pair(pts, scale, centroid)
    test_model_info_effective_scale(scale)
    print("ALL PASS")


if __name__ == "__main__":
    main()

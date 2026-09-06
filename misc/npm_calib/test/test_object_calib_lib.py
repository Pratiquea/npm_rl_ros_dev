#!/usr/bin/env python3
"""
Offline tests for npm_calib.object_calib_lib. No ROS master, no mocap.

Example:
    python3 -m pytest src/npm_rl_ros/misc/npm_calib/test/test_object_calib_lib.py
"""
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from npm_calib import object_calib_lib as ocl


def _sample_T():
    return ocl.se3_from_xyz_rpy([0.12, -0.34, 0.56], [11.0, -23.0, 47.0])


class TestConversions(unittest.TestCase):

    def test_xyz_rpy_round_trip(self):
        xyz_in, rpy_in = [0.1, -0.2, 0.3], [10.0, -20.0, 30.0]
        xyz, rpy = ocl.xyz_rpy_from_se3(ocl.se3_from_xyz_rpy(xyz_in, rpy_in))
        np.testing.assert_allclose(xyz, xyz_in, atol=1e-12)
        np.testing.assert_allclose(rpy, rpy_in, atol=1e-9)

    def test_quat_rpy_round_trip(self):
        rpy_in = [-5.0, 65.0, 120.0]
        q = ocl.rpy_to_quat_wxyz(*rpy_in)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=12)
        np.testing.assert_allclose(ocl.quat_wxyz_to_rpy(q), rpy_in, atol=1e-9)

    def test_rpy_is_fixed_axis(self):
        # Fixed-axis (extrinsic) means yaw is applied about the PARENT z, so a
        # yaw-only transform must map parent x onto parent y at 90 deg.
        T = ocl.se3_from_xyz_rpy([0, 0, 0], [0.0, 0.0, 90.0])
        np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_se3_inv_is_inverse(self):
        T = _sample_T()
        np.testing.assert_allclose(ocl.se3_inv(T) @ T, np.eye(4), atol=1e-12)


class TestSnap90(unittest.TestCase):

    def test_snaps_small_perturbation_to_exact_multiple(self):
        exact = Rotation.from_euler("xyz", [0.0, 0.0, 90.0], degrees=True)
        perturb = Rotation.from_rotvec(np.radians(3.0) * np.array([0.3, -0.5, 0.8])
                                       / np.linalg.norm([0.3, -0.5, 0.8]))
        T = ocl.make_se3((perturb * exact).as_matrix(), [0.1, 0.2, 0.3])

        snapped = ocl.snap_rotation_to_90(T)
        np.testing.assert_allclose(snapped[:3, :3], exact.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(snapped[:3, 3], [0.1, 0.2, 0.3], atol=1e-12)

    def test_snap_is_idempotent_and_orthonormal(self):
        T = ocl.snap_rotation_to_90(_sample_T())
        np.testing.assert_allclose(ocl.snap_rotation_to_90(T), T, atol=1e-12)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)

    def test_cube_group_has_24_proper_rotations(self):
        self.assertEqual(len(ocl._CUBE_GROUP), 24)


class TestResidualSplit(unittest.TestCase):
    """The gimbal-lock fix: the operator's RPY handles drive R_coarse.T @ R."""

    # The parallelopiped calibration that triggered the bug. Its absolute RPY is
    # exactly [-90, 90, 0], sitting on the pitch singularity.
    LOCKED_WXYZ = [0.5, -0.5, 0.5, 0.5]

    def _locked_T(self):
        return ocl.make_se3(ocl.quat_wxyz_to_rotmat(self.LOCKED_WXYZ),
                            [-0.005, 0.0, -0.005])

    def test_locked_pose_is_actually_on_the_singularity(self):
        # Guards the premise: if this stops holding, the rest of the class is
        # testing nothing.
        _, rpy = ocl.xyz_rpy_from_se3(self._locked_T())
        self.assertAlmostEqual(abs(rpy[1]), 90.0, places=9)

    def test_locked_pose_has_zero_residual(self):
        T = self._locked_T()
        coarse = ocl.nearest_cube_rotation(T[:3, :3])
        xyz, rpy = ocl.residual_rpy_from_se3(T, coarse)
        np.testing.assert_allclose(rpy, [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(xyz, [-0.005, 0.0, -0.005], atol=1e-12)

    def test_round_trip_over_every_cube_element_and_perturbation(self):
        rng = np.random.default_rng(0)
        for C in ocl._CUBE_GROUP:
            axis = rng.normal(size=3)
            perturb = Rotation.from_rotvec(
                np.radians(7.0) * axis / np.linalg.norm(axis)).as_matrix()
            T = ocl.make_se3(C @ perturb, rng.normal(size=3))

            coarse = ocl.nearest_cube_rotation(T[:3, :3])
            xyz, rpy = ocl.residual_rpy_from_se3(T, coarse)
            np.testing.assert_allclose(
                ocl.se3_from_coarse_and_residual(coarse, xyz, rpy), T, atol=1e-12)
            # A 7 deg perturbation must not move the nearest element, so the
            # residual stays small and far from pitch=+-90.
            np.testing.assert_allclose(coarse, C, atol=1e-12)
            self.assertLess(np.abs(rpy).max(), 10.0)

    def test_residual_always_fits_the_slider_range(self):
        # cfg/ObjectExtrinsic.cfg ranges roll/pitch/yaw at +-70 on the strength
        # of this bound; a clamped push would silently corrupt the extrinsic.
        for R in Rotation.random(2000, random_state=7).as_matrix():
            T = ocl.make_se3(R, np.zeros(3))
            coarse = ocl.nearest_cube_rotation(R)
            _, rpy = ocl.residual_rpy_from_se3(T, coarse)
            self.assertLessEqual(np.abs(rpy).max(), 70.0)
            self.assertLessEqual(ocl.residual_angle_deg(T, coarse),
                                 ocl.CUBE_COVERING_RADIUS_DEG)

    def test_residual_angle_is_the_geodesic_angle(self):
        C = ocl._CUBE_GROUP[5]
        for deg in (0.0, 3.0, 25.0, 61.0):
            axis = np.array([0.2, -0.7, 0.4])
            R = C @ Rotation.from_rotvec(
                np.radians(deg) * axis / np.linalg.norm(axis)).as_matrix()
            self.assertAlmostEqual(
                ocl.residual_angle_deg(ocl.make_se3(R, np.zeros(3)), C), deg,
                places=9)

    def test_snap_to_90_zeroes_the_residual(self):
        T = ocl.snap_rotation_to_90(_sample_T())
        coarse = ocl.nearest_cube_rotation(T[:3, :3])
        _, rpy = ocl.residual_rpy_from_se3(T, coarse)
        np.testing.assert_allclose(rpy, [0.0, 0.0, 0.0], atol=1e-9)

    def test_no_gimbal_lock_warning_escapes(self):
        # scipy warns on every as_euler at the singularity, and the tuner
        # decomposes at the TF rate; an unsuppressed warning floods the log.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            T = self._locked_T()
            ocl.xyz_rpy_from_se3(T)
            ocl.quat_wxyz_to_rpy(self.LOCKED_WXYZ)
            ocl.format_pose(T, ocl.nearest_cube_rotation(T[:3, :3]))
        self.assertEqual([str(w.message) for w in caught], [])

    def test_format_cube_rotation_names_the_axis_images(self):
        self.assertEqual(ocl.format_cube_rotation(np.eye(3)), "x->+x y->+y z->+z")
        yaw90 = ocl.se3_from_xyz_rpy([0, 0, 0], [0, 0, 90.0])[:3, :3]
        self.assertEqual(ocl.format_cube_rotation(yaw90), "x->+y y->-x z->+z")


class TestFloorResidual(unittest.TestCase):
    """Uses the real model geometry: bottom face at link z=0, so a flat-resting,
    correctly-calibrated object reads min_z 0 and tilt 0."""

    def setUp(self):
        # A stand-in for the Paralelopiped bottom+top faces; only z matters here.
        self.link_pts = np.array([[-0.5, -0.4, 0.0], [0.5, -0.4, 0.0],
                                  [0.5, 0.4, 0.0], [-0.5, 0.4, 0.0],
                                  [-0.1, 0.0, 0.6928], [0.9, 0.0, 0.6928]])

    def test_perfect_calibration_reads_zero(self):
        # Construct backwards from a known-good link pose (flat, on the floor)
        # so the composition inside floor_residual is what gets exercised.
        T_world_link = ocl.se3_from_xyz_rpy([1.0, -2.0, 0.0], [0.0, 0.0, 37.0])
        T_o1_link = ocl.se3_from_xyz_rpy([0.05, -0.02, -0.31], [0.0, 0.0, 90.0])
        T_world_o1 = T_world_link @ ocl.se3_inv(T_o1_link)

        res = ocl.floor_residual(self.link_pts, T_world_o1, T_o1_link)
        self.assertAlmostEqual(res["min_z"], 0.0, places=9)
        self.assertAlmostEqual(res["tilt_deg"], 0.0, places=9)
        self.assertEqual(res["n_below"], 0)
        self.assertTrue(res["flat"])

    def test_tilted_extrinsic_is_flagged(self):
        T_world_o1 = np.eye(4)
        res = ocl.floor_residual(self.link_pts, T_world_o1,
                                 ocl.se3_from_xyz_rpy([0, 0, 0], [25.0, 0.0, 0.0]))
        self.assertAlmostEqual(res["tilt_deg"], 25.0, places=9)
        self.assertFalse(res["flat"])

    def test_sunk_cloud_counted(self):
        T_world_o1 = ocl.se3_from_xyz_rpy([0, 0, 0], [0.0, 0.0, 0.0])
        res = ocl.floor_residual(self.link_pts, T_world_o1,
                                 ocl.se3_from_xyz_rpy([0, 0, -0.1], [0, 0, 0]))
        self.assertAlmostEqual(res["min_z"], -0.1, places=9)
        self.assertEqual(res["n_below"], 4)


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nested", "object_calib.yaml")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_save_load_round_trip(self):
        T = _sample_T()
        ocl.save_calib(self.path, T, calibrated=True, note="unit test")

        loaded, meta = ocl.load_calib(self.path)
        np.testing.assert_allclose(loaded, T, atol=1e-12)
        self.assertTrue(meta["calibrated"])
        self.assertFalse(meta["missing"])
        self.assertEqual(meta["parent_frame"], ocl.DEFAULT_PARENT)
        self.assertEqual(meta["child_frame"], ocl.DEFAULT_CHILD)

    def test_missing_file_yields_identity(self):
        T, meta = ocl.load_calib(os.path.join(self.tmp, "nope.yaml"))
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)
        self.assertTrue(meta["missing"])
        self.assertFalse(meta["calibrated"])

    def test_dict_translation_also_loads(self):
        # save_calib writes a list, matching spot_calib; a hand-edited dict form
        # must still load.
        path = os.path.join(self.tmp, "dict_form.yaml")
        with open(path, "w") as fh:
            fh.write("object_extrinsic:\n"
                     "  parent_frame: object_1\n"
                     "  child_frame: object_link\n"
                     "  translation: {x: 1.0, y: 2.0, z: 3.0}\n"
                     "  rotation_wxyz: [1.0, 0.0, 0.0, 0.0]\n")
        T, _ = ocl.load_calib(path)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)


class TestFrameMismatch(unittest.TestCase):

    def test_matching_frames_pass(self):
        meta = {"parent_frame": "parallelopiped", "child_frame": "object_link",
                "missing": False}
        self.assertIsNone(ocl.frame_mismatch(meta, "parallelopiped", "object_link"))

    def test_wrong_parent_is_reported(self):
        meta = {"parent_frame": "object_1", "child_frame": "object_link",
                "missing": False}
        msg = ocl.frame_mismatch(meta, "parallelopiped", "object_link")
        self.assertIn("parent_frame", msg)
        self.assertIn("object_1", msg)

    def test_missing_file_is_not_a_mismatch(self):
        # An absent calibration is already reported as uncalibrated; flagging it
        # as a mismatch too would bury the actionable message.
        _, meta = ocl.load_calib("/nonexistent/calib.yaml")
        self.assertIsNone(ocl.frame_mismatch(meta, "parallelopiped", "object_link"))

    def test_legacy_file_without_frame_keys_passes(self):
        self.assertIsNone(ocl.frame_mismatch({"missing": False}, "a", "b"))


if __name__ == "__main__":
    unittest.main()

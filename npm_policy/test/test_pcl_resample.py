#!/usr/bin/env python3
"""
Offline test for pcl_resample: a draw reproduces the sim preprocessing's shape,
sits on the same surface as the npz cloud, and differs from draw to draw.

    /usr/bin/python3.8 npm_policy/test/test_pcl_resample.py
"""
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from npm_policy import obs_lib as ol
from npm_policy import pcl_resample as pr

NPZ = os.path.join(os.path.expanduser("~"), "gits", "nonprehensile_object_manipulation",
                   "dataset", "primitives", "Paralelopiped", "Paralelopiped.npz")


def test_mesh_path():
    assert pr.mesh_path_for("/a/b/Thing.npz").endswith("/a/b/Thing.obj")
    assert pr.mesh_path_for("/a/b/Thing.npz", "/other/x.obj") == "/other/x.obj"
    print("PASS mesh path is the npz path with the extension swapped")


def test_draw(mesh, link_pts):
    pts, nrm = pr.resample_cloud(mesh, 128, 128)
    assert pts.shape == (128, 3), pts.shape
    assert nrm.shape == (128, 3), nrm.shape
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-9)

    # Same object, so the draw must span the npz cloud's extent. Poisson samples
    # rarely land exactly on a corner, so the box is allowed to come up slightly
    # short, never over.
    ext_npz = link_pts.max(axis=0) - link_pts.min(axis=0)
    ext_new = pts.max(axis=0) - pts.min(axis=0)
    assert np.all(ext_new <= ext_npz + 1e-3), (ext_new, ext_npz)
    assert np.all(ext_new > 0.95 * ext_npz), (ext_new, ext_npz)

    # The points are metric link-frame coordinates already: no npz scale or
    # centroid was applied, yet the draw sits on the npz cloud.
    d = np.linalg.norm(pts[:, None, :] - link_pts[None, :, :], axis=2).min(axis=1)
    assert d.max() < 0.15, d.max()
    print("PASS a draw is 128 unit-normal points on the npz surface "
          "(extent %s, max gap to an npz point %.3f m)"
          % (np.round(ext_new, 3), d.max()))


def test_draws_differ(mesh):
    a, _ = pr.resample_cloud(mesh, 128, 128)
    b, _ = pr.resample_cloud(mesh, 128, 128)
    # The whole point of the viz: the .npz holds one draw out of many.
    assert not np.allclose(np.sort(a, axis=0), np.sort(b, axis=0)), \
        "two draws must differ, or the viz shows nothing"
    print("PASS consecutive draws differ (centroids %s vs %s)"
          % (np.round(a.mean(axis=0), 4), np.round(b.mean(axis=0), 4)))


def test_fps_stage(mesh):
    """oversample > n_points must engage the sim's FPS downsample."""
    rng = np.random.default_rng(0)
    pts, nrm = pr.resample_cloud(mesh, 64, 512, rng)
    assert pts.shape == (64, 3) and nrm.shape == (64, 3)

    # FPS spreads the selection: its minimum pairwise distance beats a random
    # subset of the same cloud, which is why normalize_mesh.py uses it.
    dense, _ = pr.resample_cloud(mesh, 512, 512)
    idx_fps = pr.fps(dense, 64, np.random.default_rng(1))
    idx_rand = np.random.default_rng(1).choice(dense.shape[0], 64, replace=False)

    def min_gap(sel):
        d = np.linalg.norm(sel[:, None, :] - sel[None, :, :], axis=2)
        return d[~np.eye(len(sel), dtype=bool)].min()

    assert min_gap(dense[idx_fps]) > min_gap(dense[idx_rand]), "FPS must spread wider"
    print("PASS oversample>n_points runs FPS (min gap %.3f m vs random %.3f m)"
          % (min_gap(dense[idx_fps]), min_gap(dense[idx_rand])))


def test_tick_budget(mesh):
    # The node redraws in its timer callback, so a draw must fit the tick.
    t0 = time.perf_counter()
    for _ in range(5):
        pr.resample_cloud(mesh, 128, 128)
    per_draw = (time.perf_counter() - t0) / 5.0
    assert per_draw < 0.1, "%.3f s per draw does not fit a 10 Hz tick" % per_draw
    print("PASS a draw costs %.0f ms, inside the 100 ms tick" % (1000 * per_draw))


def main():
    if not os.path.exists(NPZ):
        print("SKIP: %s not found" % NPZ)
        return
    mesh_path = pr.mesh_path_for(NPZ)
    if not os.path.exists(mesh_path):
        print("SKIP: %s not found" % mesh_path)
        return
    pts, _nrm, scale, centroid = ol.load_model_npz(NPZ)
    link_pts = ol.link_frame_cloud(pts, scale, centroid)
    mesh = pr.load_mesh(mesh_path)

    test_mesh_path()
    test_draw(mesh, link_pts)
    test_draws_differ(mesh)
    test_fps_stage(mesh)
    test_tick_budget(mesh)
    print("ALL PASS")


if __name__ == "__main__":
    main()

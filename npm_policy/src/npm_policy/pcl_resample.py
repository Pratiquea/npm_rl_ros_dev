#!/usr/bin/env python3
"""
Redraw the object point cloud from its source mesh, the way the sim's
utils/normalize_mesh.py --method poisson does (the step run_preprocessing.py
runs to build the .npz).

VISUAL ONLY. Nothing here may reach the observation, the pointer argmax, or a
push point. No ROS imports, so it is unit-testable offline; open3d is imported
lazily so the rest of npm_policy still works without it.
"""
import os

import numpy as np


def mesh_path_for(npz_path, override=""):
    # run_preprocessing.py writes <name>.obj and <name>.npz side by side, so the
    # mesh is the npz path with the extension swapped.
    if override:
        return override
    return os.path.splitext(str(npz_path))[0] + ".obj"


def load_mesh(path):
    """Mesh with the orientation and normals normalize_mesh.py computes.

    orient_triangles() before the normal computation is not cosmetic: it is what
    makes use_triangle_normal below yield outward normals, which the cone action
    decode depends on being outward.
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError("empty or invalid mesh %s" % (path,))
    mesh.orient_triangles()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def fps(points, n_samples, rng):
    # Farthest Point Sampling, ported verbatim from utils/normalize_mesh.py so a
    # resample at the npz's oversample reproduces its point distribution.
    pts = np.asarray(points, dtype=np.float64)
    n_total = pts.shape[0]
    n_samples = min(int(n_samples), n_total)
    idxs = np.zeros(n_samples, dtype=np.int64)
    idxs[0] = rng.integers(0, n_total)
    d2 = np.sum((pts - pts[idxs[0]]) ** 2, axis=1)
    for i in range(1, n_samples):
        idxs[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, np.sum((pts - pts[idxs[i]]) ** 2, axis=1))
    return idxs


def resample_cloud(mesh, n_points=128, oversample=128, rng=None):
    """(points (n,3), normals (n,3)) in the mesh (== link) frame, in metres.

    Poisson-disk sample, then FPS down to n_points when oversample exceeds it.
    The npz was built with oversample = max(n_points*8, 10000), which costs ~1.4 s
    a draw; oversample == n_points skips the FPS stage and runs in ~16 ms, which
    is what makes a live redraw possible at all.

    No normalize/denormalize: that round trip is the identity, so these points are
    already the link-frame cloud. They are NOT the npz's points - a fresh draw has
    its own centroid, a few mm off the npz one, and that difference is the thing
    worth looking at.
    """
    over = max(int(oversample), int(n_points))
    # open3d exposes no seed here, so every call differs. That is what animates
    # the draw; it also means a resample cannot be made reproducible.
    pcd = mesh.sample_points_poisson_disk(over, use_triangle_normal=True)
    pts = np.asarray(pcd.points, dtype=np.float64)
    nrm = np.asarray(pcd.normals, dtype=np.float64)

    if over > int(n_points):
        idx = fps(pts, n_points, rng if rng is not None else np.random.default_rng())
        pts, nrm = pts[idx], nrm[idx]

    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pts, nrm

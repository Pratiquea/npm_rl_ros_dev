#!/usr/bin/env python3
import os

import numpy as np


def mesh_path_for(npz_path, override=""):
    if override:
        return override
    return os.path.splitext(str(npz_path))[0] + ".obj"


def load_mesh(path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError("empty or invalid mesh %s" % (path,))
    mesh.orient_triangles()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def fps(points, n_samples, rng):
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
    over = max(int(oversample), int(n_points))
    pcd = mesh.sample_points_poisson_disk(over, use_triangle_normal=True)
    pts = np.asarray(pcd.points, dtype=np.float64)
    nrm = np.asarray(pcd.normals, dtype=np.float64)

    if over > int(n_points):
        idx = fps(pts, n_points, rng if rng is not None else np.random.default_rng())
        pts, nrm = pts[idx], nrm[idx]

    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pts, nrm


def mesh_triangles(mesh):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if verts.size == 0 or tris.size == 0:
        raise ValueError("mesh has no triangles")
    return verts, tris

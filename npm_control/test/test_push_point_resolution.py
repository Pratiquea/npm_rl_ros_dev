#!/usr/bin/env python3
"""
Offline test for el.resolve_push_point: the loc_idx/push_point sentinel and
executor-vs-policy agreement on what a loc_idx means.

    python3 npm_control/test/test_push_point_resolution.py
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "npm_policy", "src"))
from npm_control import executor_lib as el
from npm_policy import obs_lib as ol

_NPZ = os.path.join(os.path.expanduser("~"), "gits", "nonprehensile_object_manipulation",
                    "dataset", "primitives", "Paralelopiped", "Paralelopiped.npz")


class _Point(object):
    # Stand-in for geometry_msgs/Point so the test needs no ROS.
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def main():
    # --- sentinel: loc_idx < 0 hands back the explicit point untouched ---
    link_pts = np.arange(128 * 3, dtype=np.float64).reshape(128, 3)
    pt, src = el.resolve_push_point(-1, _Point(0.1, 0.2, 0.3), link_pts)
    assert src == "push_point", src
    assert np.allclose(pt, [0.1, 0.2, 0.3]), pt
    # a cloud is not even required on the bench path
    pt, src = el.resolve_push_point(-5, [0.0, 0.0, 0.0], None)
    assert src == "push_point" and np.allclose(pt, [0.0, 0.0, 0.0])

    # --- loc_idx indexes the cloud, and push_point is ignored ---
    pt, src = el.resolve_push_point(7, _Point(9.0, 9.0, 9.0), link_pts)
    assert src == "loc_idx", src
    assert np.allclose(pt, link_pts[7]), pt
    pt[0] = -1.0
    assert link_pts[7][0] == 21.0, "resolve_push_point must not alias the cloud"

    # --- refuse rather than substitute: a wrong point pushes the wrong face ---
    for bad_args in ((0, _Point(0, 0, 0), None), (128, _Point(0, 0, 0), link_pts)):
        try:
            el.resolve_push_point(*bad_args)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (bad_args[0],))
    print("PASS: sentinel selects the path; loc_idx indexes the cloud; bad indices raise")

    # --- parity with the policy on the real model ---
    if not os.path.exists(_NPZ):
        print("SKIP: %s not found, parity check not run" % _NPZ)
        return
    pts, _normals, scale, centroid = ol.load_model_npz(_NPZ)
    # Both shrink values: the executor and the policy must agree on what a loc_idx
    # means for whatever pcl_shrink the stack is running.
    for shrink in (1.0, 0.97):
        cloud = ol.link_frame_cloud(pts, scale, centroid, shrink)
        for loc in (0, 1, 63, 127):
            got, src = el.resolve_push_point(loc, _Point(0, 0, 0), cloud)
            # policy_lib.postprocess computes contact_link with this exact
            # expression, so matching it is the executor-vs-policy agreement the
            # runtime warn guards.
            want = ol.link_frame_cloud(pts, scale, centroid, shrink)[loc]
            assert src == "loc_idx"
            assert np.allclose(got, want, atol=1e-12), (shrink, loc, got, want)

    # A shrink mismatch really does move the commanded point, which is why
    # /npm/model_info carries the effective scale.
    nominal = ol.link_frame_cloud(pts, scale, centroid)
    shrunk = ol.link_frame_cloud(pts, scale, centroid, 0.97)
    assert np.linalg.norm(nominal[0] - shrunk[0]) > 1e-3
    print("PASS: executor resolution matches the policy's contact_link for %s "
          "at pcl_shrink 1.0 and 0.97" % os.path.basename(_NPZ))


if __name__ == "__main__":
    main()

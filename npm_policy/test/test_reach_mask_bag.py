#!/usr/bin/env python3
"""
Replay hand_push_policy.bag through the deploy inference chain and check that the
reachability gate fixes the livelock that bag recorded, without disturbing the
macro-steps that were already fine.

That run stalled on macro-steps 5-20: the policy picked loc_idx=5 at world
z=0.029, the push could not happen, the object did not move, so the observation
froze and the deterministic argmax returned 5 sixteen times over 44 s.

Two claims, in order, because the second is worthless without the first:
  1. with the gate off, the replay reproduces the loc_idx the bag recorded, so
     the harness is faithful;
  2. with the gate on, every contact point clears the ground clearance, and the
     macro-steps that were already reachable keep the exact index they had.

Skips rather than fails when the bag, the checkpoint or rosbag is absent, so it
is safe to run on the robot.

Example:
    source /opt/ros/noetic/setup.bash && /usr/bin/python3 test_reach_mask_bag.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from npm_policy import policy_lib as pl
from npm_policy import obs_lib as ol

BAG = os.environ.get("BAG", "/media/ssd/rosbags/npm/hand_push_policy.bag")
GITS = "/home/rwl-4090/gits/nonprehensile_object_manipulation"
CKPT = os.environ.get(
    "CKPT", GITS + "/logs/rsl_rl/object_manip_discrete_direct/2026-09-01_00-50-05/model_1100.pt")
NPZ = os.environ.get("NPZ", GITS + "/dataset/primitives/Paralelopiped/Paralelopiped.npz")

# npm.yaml at the time of the run.
MASS, FRICTION = 10.0, 0.95

# The bag starts mid-episode, so the prev_action that fed macro-step 1 was never
# recorded and the replay cannot reproduce it. Every later step chains from a
# recorded one.
FIRST_REPRODUCIBLE = 2


def main():
    try:
        import rosbag
    except ImportError:
        print("reach_mask_bag SKIPPED: rosbag not importable (source ROS first)")
        return
    for path, what in ((BAG, "bag"), (CKPT, "checkpoint"), (NPZ, "npz")):
        if not os.path.isfile(path):
            print("reach_mask_bag SKIPPED: %s not found: %s" % (what, path))
            return

    states, bag_locs = [], []
    with rosbag.Bag(BAG) as b:
        for _t, m, _s in b.read_messages(topics=["/npm/served_object_state"]):
            states.append(m)
        for _t, m, _s in b.read_messages(topics=["/npm/push_command"]):
            bag_locs.append(int(m.loc_idx))
    assert len(states) == len(bag_locs), (len(states), len(bag_locs))
    assert len(states) > 0, "bag has no served_object_state"

    bundle = pl.PolicyBundle(CKPT, NPZ, MASS, FRICTION)
    prev = np.zeros(4)
    n_repro, n_moved, n_inert, n_stuck = 0, 0, 0, 0

    for i, s in enumerate(states):
        seq = i + 1
        rot = ol.rotation_from_ros_quat(s.pose.orientation)
        pos = np.array([s.pose.position.x, s.pose.position.y, s.pose.position.z])
        lv = np.array([s.lin_vel.x, s.lin_vel.y, s.lin_vel.z])
        av = np.array([s.ang_vel.x, s.ang_vel.y, s.ang_vel.z])

        out_u = pl.infer_push(bundle, rot, pos, lv, av, prev, mask_reach=False)
        out_m = pl.infer_push(bundle, rot, pos, lv, av, prev, mask_reach=True)
        loc_u, cw_u, next_prev = int(out_u[2]), out_u[6], out_u[7]
        loc_m, cw_m = int(out_m[2]), out_m[6]
        reach = pl.reachable_mask(rot, pos, bundle.link_pts, bundle.normals)

        # Claim 1: the unmasked chain is the run that was recorded.
        if seq >= FIRST_REPRODUCIBLE:
            assert loc_u == bag_locs[i], \
                "seq %d: replay picked %d, bag recorded %d" % (seq, loc_u, bag_locs[i])
            n_repro += 1

        # Claim 2a: the gate never emits a point at floor level.
        assert reach[loc_m], "seq %d: masked loc=%d is unreachable" % (seq, loc_m)
        assert cw_m[2] > pl.GROUND_CLEARANCE, \
            "seq %d: masked contact z=%.4f below clearance" % (seq, cw_m[2])

        # Claim 2b: on a step that was already reachable the gate is inert. A gate
        # that also reroutes the good steps is changing the policy, not fixing it.
        if seq >= FIRST_REPRODUCIBLE:
            if cw_u[2] > pl.GROUND_CLEARANCE:
                assert loc_m == loc_u, \
                    "seq %d: gate moved an already-reachable pick %d -> %d" \
                    % (seq, loc_u, loc_m)
                n_inert += 1
            else:
                assert loc_m != loc_u, \
                    "seq %d: gate left the unreachable pick %d in place" % (seq, loc_u)
                n_moved += 1
                n_stuck += 1

        # The observation is a function of the previous DECODED action, so the
        # chain has to advance on the unmasked branch to keep reproducing the bag.
        prev = next_prev

    assert n_stuck > 0, "bag has no unreachable pick; it cannot prove anything"
    print("PASS: %d/%d macro-steps reproduced the bag; gate rerouted %d "
          "floor-level picks and left %d reachable picks untouched"
          % (n_repro, len(states) - FIRST_REPRODUCIBLE + 1, n_moved, n_inert))


if __name__ == "__main__":
    main()

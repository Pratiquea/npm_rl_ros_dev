#!/usr/bin/env python3
"""check_retreat_bag: offline acceptance check for the executor's RETREAT phase.

Replays a bag's /tf into a tf2 buffer and measures, over the RETREAT window, whether
the arm actually retracted or the body walked into the arm. Written against the
inverted-retreat bug found in spot_push_7.bag (see wiki/log.md 2026-08-20); the
criteria are the direct inverse of the four measurements that characterised it.

Needs /tf, /tf_static, /npm/executor/phase and /npm/ee_force in the bag.
/npm/executor/assist_debug is optional; when present, the last line the executor
published before the retreat is echoed, which says why a body assist did or did not
fire (and so why the push ended where it did).

    rosrun npm_control check_retreat_bag.py /media/ssd/rosbags/npm/spot_push_N.bag
"""
import argparse
import sys

import numpy as np
import rosbag
import rospy
import tf2_ros

PHASE_PUSH = 2
PHASE_RETREAT = 3


def _load(bag_path, root, object_frame):
    """Replay the bag into a tf buffer; return (buffer, phases, forces, assists)."""
    # Buffer the whole bag: cache_time must cover the run or lookups fall off the back.
    buf = tf2_ros.Buffer(rospy.Duration(3600), debug=False)
    phases, forces, assists = [], [], []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(
                topics=["/tf", "/tf_static", "/npm/executor/phase", "/npm/ee_force",
                        "/npm/executor/assist_debug"]):
            if topic == "/tf":
                for tr in msg.transforms:
                    buf.set_transform(tr, "bag")
            elif topic == "/tf_static":
                for tr in msg.transforms:
                    buf.set_transform_static(tr, "bag")
            elif topic == "/npm/executor/phase":
                phases.append((t.to_sec(), int(msg.data)))
            elif topic == "/npm/executor/assist_debug":
                assists.append((t.to_sec(), str(msg.data)))
            else:
                forces.append((t.to_sec(), float(msg.data)))
    return buf, phases, forces, assists


def _window(phases, phase_id):
    """[start, end] wall times of the last contiguous run of phase_id."""
    hits = [ts for ts, v in phases if v == phase_id]
    if not hits:
        return None
    return hits[0], hits[-1]


def _xyz(buf, target, source, ts):
    tr = buf.lookup_transform(target, source, rospy.Time(ts))
    p = tr.transform.translation
    return np.array([p.x, p.y, p.z])


def _push_dir(buf, phases, root, object_frame):
    """Push direction in root, taken from how the object moved during PUSH."""
    win = _window(phases, PHASE_PUSH)
    if win is None:
        return None
    a = _xyz(buf, root, object_frame, win[0])
    b = _xyz(buf, root, object_frame, win[1])
    d = b - a
    n = np.linalg.norm(d[:2])
    if n < 1e-6:
        return None
    return np.array([d[0] / n, d[1] / n, 0.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--root", default="odom")
    # The link frame, matching what the executor tracks; the raw mocap body
    # frame moves with the Motive pivot, not the pushed point.
    ap.add_argument("--object-frame", default="object_link", dest="object_frame")
    ap.add_argument("--body-frame", default="body", dest="body_frame")
    ap.add_argument("--hand-frame", default="hand", dest="hand_frame")
    ap.add_argument("--min-pullback", type=float, default=0.15, dest="min_pullback",
                    help="m of hand travel AGAINST push_dir required (criterion 1)")
    ap.add_argument("--max-object-move", type=float, default=0.02, dest="max_object_move",
                    help="m the object may move during RETREAT (criterion 3)")
    ap.add_argument("--contact-eps", type=float, default=2.0, dest="contact_eps",
                    help="N the EE force must fall below (criterion 4)")
    cfg = ap.parse_args(sys.argv[1:])

    buf, phases, forces, assists = _load(cfg.bag, cfg.root, cfg.object_frame)
    win = _window(phases, PHASE_RETREAT)
    if win is None:
        print("FAIL: no PHASE_RETREAT (3) on /npm/executor/phase")
        return 1
    t_a, t_b = win
    d = _push_dir(buf, phases, cfg.root, cfg.object_frame)
    if d is None:
        print("FAIL: could not derive push_dir from object motion during PUSH")
        return 1

    body_a, body_b = (_xyz(buf, cfg.root, cfg.body_frame, t) for t in (t_a, t_b))
    hand_a, hand_b = (_xyz(buf, cfg.root, cfg.hand_frame, t) for t in (t_a, t_b))
    obj_a, obj_b = (_xyz(buf, cfg.root, cfg.object_frame, t) for t in (t_a, t_b))

    hand_along = float(np.dot(hand_b - hand_a, d))
    body_along = float(np.dot(body_b - body_a, d))
    obj_move = float(np.linalg.norm(obj_b - obj_a))
    body_obj_a = float(np.linalg.norm((body_a - obj_a)[:2]))
    body_obj_b = float(np.linalg.norm((body_b - obj_b)[:2]))
    # settled force = last quarter of the window, after the RELEASE ramp
    tail = [f for ts, f in forces if t_b - 0.25 * (t_b - t_a) <= ts <= t_b]
    force_tail = max(tail) if tail else float("nan")

    print("RETREAT window %.2f -> %.2f s (%.2f s), push_dir=(%.3f, %.3f)"
          % (t_a, t_b, t_b - t_a, d[0], d[1]))
    print("  hand along push_dir : %+.3f m   (want <= %+.3f)"
          % (hand_along, -cfg.min_pullback))
    print("  body along push_dir : %+.3f m" % body_along)
    print("  body-object dist    : %.3f -> %.3f m" % (body_obj_a, body_obj_b))
    print("  object moved        : %.3f m   (want < %.3f)" % (obj_move, cfg.max_object_move))
    print("  peak force in tail  : %.1f N   (want < %.1f)" % (force_tail, cfg.contact_eps))
    # Last assist verdict before the retreat: why the push ended where it did. Not a
    # pass/fail criterion, it is the context for the four that follow.
    pre = [(ts, line) for ts, line in assists if ts <= t_a]
    if pre:
        print("  last assist verdict : %s" % pre[-1][1])
    elif assists:
        print("  last assist verdict : (none before retreat; %d lines in bag)"
              % len(assists))
    else:
        print("  last assist verdict : (no /npm/executor/assist_debug in bag)")

    checks = [
        ("1 hand retracted", hand_along <= -cfg.min_pullback),
        ("2 body did not close on object", body_obj_b >= body_obj_a - 1e-3),
        ("3 object stayed put", obj_move < cfg.max_object_move),
        ("4 force released", not np.isnan(force_tail) and force_tail < cfg.contact_eps),
    ]
    print()
    for name, ok in checks:
        print("  %s %s" % ("PASS" if ok else "FAIL", name))
    failed = [n for n, ok in checks if not ok]
    print("\n%s" % ("ALL PASS" if not failed else "FAILED: " + ", ".join(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

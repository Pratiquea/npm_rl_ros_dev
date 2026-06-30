#!/usr/bin/env python3
"""calibrate_from_bag: read the rosbag from data_collection_node and solves the robot-world
   hand-eye problem, and writes a yaml file holding both extrinsics + residuals.

   rosrun spot_calib calibrate_from_bag.py \
      _bag:=/media/ssd/rosbags/npm/spot_calib.bag \
      _out_yaml:=$(rospack find spot_calib)/config/spot_mocap_calib.yaml

"""
import argparse
import numpy as np
import yaml

import rosbag
from spot_calib import calib_lib as cl


def _msg_to_se3(msg):
    t = msg.transform.translation
    r = msg.transform.rotation
    R = cl.quat_wxyz_to_rotmat([r.w, r.x, r.y, r.z])
    return cl.make_se3(R, [t.x, t.y, t.z])


def _odom_to_se3(msg):
    """nav_msgs/Odometry pose to SE3 ."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    R = cl.quat_wxyz_to_rotmat([q.w, q.x, q.y, q.z])
    return cl.make_se3(R, [p.x, p.y, p.z])


def _find_static_tf(bag_path, parent, child):
    """Return the constant SE3 T_{parent<-child} from /tf(_static), or None."""
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=["/tf", "/tf_static"]):
            for tr in msg.transforms:
                if tr.header.frame_id == parent and tr.child_frame_id == child:
                    return _msg_to_se3(tr)
    return None


def load_raw(bag_path, mocap_raw_topic, spot_raw_topic):
    """Load the dense raw odometry streams for offline re-pairing.

    Returns (tA, A, tB, B): mocap A is composed into the calibration `world`
    frame (raw Odom is mocap_world<-spot_body_mocap; left-multiply by the
    constant world<-mocap_world). Spot B (vision<-body) already equals the
    calib frame because body==base_link is identity in /tf_static."""
    tA, A_mw, tB, B = [], [], [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, _ in bag.read_messages(
                topics=[mocap_raw_topic, spot_raw_topic]):
            s = msg.header.stamp.to_sec()
            if topic == mocap_raw_topic:
                tA.append(s); A_mw.append(_odom_to_se3(msg))
            else:
                tB.append(s); B.append(_odom_to_se3(msg))
    Twm = _find_static_tf(bag_path, "world", "mocap_world")
    if Twm is None:
        raise SystemExit("ERROR: --td_source raw needs the world->mocap_world "
                         "TF to bring raw mocap into the calib `world` frame; "
                         "not found in bag.")
    A = [Twm @ a for a in A_mw]
    return tA, A, tB, B


def load_pairs(bag_path):
    A, B = [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=["/calib/A", "/calib/B"]):
            (A if topic == "/calib/A" else B).append(_msg_to_se3(msg))
    n = min(len(A), len(B))
    if len(A) != len(B):
        print("WARN: unequal counts A=%d B=%d, truncating to %d" %
              (len(A), len(B), n))
    return A[:n], B[:n]


def load_stamped(bag_path):
    """Return (tA, A, tB, B)."""
    tA, A, tB, B = [], [], [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=["/calib/A", "/calib/B"]):
            s = msg.header.stamp.to_sec()
            if topic == "/calib/A":
                tA.append(s); A.append(_msg_to_se3(msg))
            else:
                tB.append(s); B.append(_msg_to_se3(msg))
    return tA, A, tB, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--time-align", choices=["none", "grid", "xcorr"],
                    default="xcorr",
                    help="offline removal of the mocap<->spot time offset td")
    ap.add_argument("--max_td", type=float, default=0.6,
                    help="max td search range (s) for time-align")
    ap.add_argument("--dt", type=float, default=0.005,
                    help="time-align search step (s)")
    ap.add_argument("--td_source", choices=["calib", "raw"], default="calib",
                    help="estimate td and build pairs from the throttled "
                         "/calib topics (calib) or the dense raw odometry "
                         "streams (raw)")
    ap.add_argument("--mocap_raw_topic", default="/mocap_node/spot_body/Odom",
                    help="raw mocap odometry topic for --td_source raw")
    ap.add_argument("--spot_raw_topic", default="/spot/odometry_corrected",
                    help="raw spot odometry topic for --td_source raw")
    ap.add_argument("--raw_pair_hz", type=float, default=10.0,
                    help="subsample the dense mocap timeline to this rate "
                         "before re-pairing (--td_source raw)")
    args, _ = ap.parse_known_args()

    try:
        import rospy
        rospy.init_node("calibrate_from_bag", anonymous=True,
                        disable_signals=True)
        bag_path = rospy.get_param("~bag", args.bag) if rospy.core.is_initialized() else args.bag
        out_yaml = rospy.get_param("~out_yaml", args.out)
        time_align = rospy.get_param("~time_align", args.time_align)
        td_source = rospy.get_param("~td_source", args.td_source)
    except Exception:
        bag_path, out_yaml, time_align = args.bag, args.out, args.time_align
        td_source = args.td_source
    bag_path = bag_path or "/tmp/spot_calib.bag"
    out_yaml = out_yaml or "/tmp/spot_extrinsic.yaml"

    tA, A, tB, B = load_stamped(bag_path)
    n = min(len(A), len(B))
    print("loaded %d pairs from %s" % (n, bag_path))
    if len(A) != len(B):
        print("WARN: unequal counts A=%d B=%d, truncating to %d" %
              (len(A), len(B), n))
    if n < 3:
        raise SystemExit("ERROR: only %d pairs in bag; need >=3." % n)

    # Pair from the dense raw odometry streams instead of the throttled
    # (resamples B onto the A timeline + td).
    if td_source == "raw":
        tAr, Ar, tBr, Br = load_raw(bag_path, args.mocap_raw_topic,
                                    args.spot_raw_topic)
        print("loaded raw: mocap %s=%d, spot %s=%d" %
              (args.mocap_raw_topic, len(Ar), args.spot_raw_topic, len(Br)))
        if len(Ar) < 3 or len(Br) < 3:
            print("WARN: raw stream too short (mocap=%d spot=%d); "
                  "falling back to td_source=calib." % (len(Ar), len(Br)))
            td_source = "calib"
        else:
            # subsample the dense mocap timeline to ~spot rate so B is not
            # highly over-interpolated during resampling
            tAr = np.asarray(tAr, float)
            order = np.argsort(tAr)
            tAr = tAr[order]; Ar = [Ar[i] for i in order]
            step = 1.0 / args.raw_pair_hz if args.raw_pair_hz > 0 else 0.0
            tA, A, last = [], [], -np.inf
            for t, a in zip(tAr, Ar):
                if t - last >= step:
                    tA.append(t); A.append(a); last = t
            tB, B = list(tBr), list(Br)
            print("raw: %d mocap samples after %.1f Hz subsample" %
                  (len(A), args.raw_pair_hz))

    # offline time alignment
    td = 0.0
    if time_align != "none" or td_source == "raw":
        if time_align == "xcorr":
            td = cl.estimate_td_xcorr(tA, A, tB, B, max_td=args.max_td, dt=args.dt)
        elif time_align == "grid":
            td, _, _ = cl.estimate_td_grid(A, tA, tB, B, td_range=(-args.max_td, args.max_td), step=args.dt)
        print("time-align(%s): td = %+.1f ms" % (time_align, td * 1e3))
        qt, Bres = cl.resample_stream(tB, B, np.asarray(tA, float) + td)
        keep = {round(float(t) - td, 9): k for k, t in enumerate(qt)}
        A2, B2 = [], []
        for k, t in enumerate(tA):
            j = keep.get(round(float(t), 9))
            if j is not None:
                A2.append(A[k]); B2.append(Bres[j])
        if len(A2) < 3:
            raise SystemExit("ERROR: <3 pairs survived resampling at td=%.3f" % td)
        A, B = A2, B2
        print("time-align: %d pairs after resampling" % len(A))
    else:
        A, B = A[:n], B[:n]

    m = cl.excitation(A)
    print("excitation:", m)
    if not m["ready"]:
        print("ERROR: motion is (near-)degenerate "
              "(n=%d, l2r=%.3f). Re-collect with rotations about >=2 "
              "non-parallel axes (yaw + pitch/roll). Aborting." %
              (m["n"], m["l2r"]))
        raise SystemExit(2)

    Xb, Y = cl.solve_robot_world(A, B)
    rot_err, trans_err = cl.residuals(A, B, Xb, Y)
    print("residual: %.4f deg, %.4f m" % (rot_err, trans_err))

    # cross-check with the manual AX=XB solver
    Xb_chk = cl.solve_axxb_manual(A, B)
    dq = np.degrees(np.linalg.norm(
        cl.rotmat_to_rotvec((cl.se3_inv(Xb) @ Xb_chk)[:3, :3])))
    dt = np.linalg.norm(Xb[:3, 3] - Xb_chk[:3, 3])
    print("cv2-vs-manual disagreement: %.3f deg, %.4f m" % (dq, dt))

    def block(T, parent, child):
        q = cl.rotmat_to_quat_wxyz(T[:3, :3])
        return dict(parent_frame=parent, child_frame=child,
                    translation=[float(v) for v in T[:3, 3]],
                    rotation_wxyz=[float(v) for v in q])

    out = dict(
        # Xb: body extrinsic
        body_extrinsic=block(Xb, "spot_body_mocap", "base_link"),
        # Y: world link 
        world_link=block(Y, "world", "vision"),
        residual=dict(rot_deg=round(rot_err, 4), trans_m=round(trans_err, 4),
                      time_align=time_align, td_source=td_source,
                      td_ms=round(td * 1e3, 2)),
        excitation=m, num_pairs=len(A),
    )
    with open(out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print("wrote %s" % out_yaml)


if __name__ == "__main__":
    main()
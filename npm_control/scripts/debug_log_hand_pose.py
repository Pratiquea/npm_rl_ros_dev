#!/usr/bin/env python3
"""log_hand_pose; read-only tool to measure the current hand pose in the body
frame for whatever pose the arm is already in (stow, unstow, carry, custom...).

Motion-free: it never takes a lease and never commands the robot, so it is safe
to run while another node holds the lease. Put the arm in the pose you want
(manually, tablet, or another node), then run this to read body->hand straight
from the robot's frame_tree_snapshot -- the only trustworthy source, since the
spot_ros URDF is not kinematically matched to bosdyn.

Run example:
    rosrun npm_control log_hand_pose.py _hostname:=ROBOT_IP
    # one-shot, averaged over 10 samples. For a live readout while you jog the arm:
    rosrun npm_control log_hand_pose.py _hostname:=ROBOT_IP _watch:=true _rate:=5
"""
import sys

import numpy as np
import rospy

import bosdyn.client
import bosdyn.client.time_sync
import bosdyn.client.util
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME,
                                         GRAV_ALIGNED_BODY_FRAME_NAME,
                                         HAND_FRAME_NAME, get_a_tform_b)
from bosdyn.client.robot_state import RobotStateClient


def _quat_wxyz(q):
    # bosdyn SE3Pose.rotation is a Quaternion with .w .x .y .z fields.
    return np.array([q.w, q.x, q.y, q.z])


def _read(state_client, body_frame):
    st = state_client.get_robot_state()
    snap = st.kinematic_state.transforms_snapshot
    tf = get_a_tform_b(snap, body_frame, HAND_FRAME_NAME)
    if tf is None:
        raise RuntimeError("hand transform missing from snapshot (arm attached?)")
    return tf, st.kinematic_state.joint_states


def _arm_joint_angles(joint_states):
    # Arm joints are named arm0.sh0, arm0.sh1, arm0.el0, arm0.el1, arm0.wr0,
    # arm0.wr1 (plus f1x gripper). Keep declared order for a stable printout.
    order = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0",
             "arm0.wr1", "arm0.f1x"]
    by_name = {j.name: j.position.value for j in joint_states}
    return [(n, by_name[n]) for n in order if n in by_name]


def _print_pose(tf_body, tf_grav, joints):
    p = tf_body.position
    q = _quat_wxyz(tf_body.rotation)
    pg = tf_grav.position
    print("body      -> hand : pos [%+.4f %+.4f %+.4f]  quat_wxyz [%+.4f %+.4f %+.4f %+.4f]"
          % (p.x, p.y, p.z, q[0], q[1], q[2], q[3]))
    print("flat_body -> hand : pos [%+.4f %+.4f %+.4f]" % (pg.x, pg.y, pg.z))
    if joints:
        print("arm joints (rad)  : " + "  ".join("%s=%+.3f" % (n, v) for n, v in joints))


def main():
    rospy.init_node("log_hand_pose", anonymous=True)
    hostname = rospy.get_param("~hostname", None)
    samples = int(rospy.get_param("~samples", 10))
    watch = bool(rospy.get_param("~watch", False))
    rate_hz = float(rospy.get_param("~rate", 5.0))
    time_sync_timeout = float(rospy.get_param("~time_sync_timeout", 15.0))
    if not hostname:
        rospy.logerr("~hostname is required (rosrun ... _hostname:=ROBOT_IP)")
        return 1

    bosdyn.client.util.setup_logging(False)
    sdk = bosdyn.client.create_standard_sdk("NpmHandPose")
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    # Reading state needs time-sync but no lease and no estop; this tool never moves.
    robot.time_sync.wait_for_sync(timeout_sec=time_sync_timeout)
    assert robot.has_arm(), "Robot requires an arm."

    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    if watch:
        # Live readout; put the arm where you want and watch body->hand update.
        print("Watching body->hand at %.1f Hz. Ctrl-C to stop.\n" % rate_hz)
        rate = rospy.Rate(max(0.1, rate_hz))
        while not rospy.is_shutdown():
            tf_b, js = _read(state_client, BODY_FRAME_NAME)
            tf_g, _ = _read(state_client, GRAV_ALIGNED_BODY_FRAME_NAME)
            _print_pose(tf_b, tf_g, _arm_joint_angles(js))
            print("-" * 60)
            rate.sleep()
        return 0

    # One-shot: average body->hand over a few reads to smooth snapshot jitter.
    pos, quat = [], []
    js_last = None
    for _ in range(max(1, samples)):
        tf_b, js_last = _read(state_client, BODY_FRAME_NAME)
        pos.append([tf_b.position.x, tf_b.position.y, tf_b.position.z])
        quat.append(_quat_wxyz(tf_b.rotation))
        rospy.sleep(0.05)
    pos = np.array(pos)
    quat = np.array(quat)
    pmean, pstd = pos.mean(0), pos.std(0)
    qmean = quat.mean(0)
    qmean /= np.linalg.norm(qmean)
    tf_g, _ = _read(state_client, GRAV_ALIGNED_BODY_FRAME_NAME)

    print("\n==== current hand pose (from frame_tree_snapshot, no motion) ====")
    print("frame: body -> hand   (meters, wxyz)")
    print("  pos  = [%+.4f %+.4f %+.4f]" % tuple(pmean))
    print("  std  = [ %.4f  %.4f  %.4f]  (n=%d)" % (pstd[0], pstd[1], pstd[2], len(pos)))
    print("  quat = [%+.4f %+.4f %+.4f %+.4f]" % tuple(qmean))
    print("frame: flat_body(grav-aligned) -> hand")
    print("  pos  = [%+.4f %+.4f %+.4f]" % (tf_g.position.x, tf_g.position.y,
                                            tf_g.position.z))
    ja = _arm_joint_angles(js_last)
    if ja:
        print("arm joints (rad):")
        print("  " + "  ".join("%s=%+.3f" % (n, v) for n, v in ja))
    print("=================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

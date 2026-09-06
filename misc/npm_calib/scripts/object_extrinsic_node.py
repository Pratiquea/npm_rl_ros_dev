#!/usr/bin/env python3
"""
Publish the calibrated <mocap body> -> object_link extrinsic on /tf_static.
Runs in the normal stack; object_calib_node is the tuner and does not.

Example:
    rosrun npm_calib object_extrinsic_node.py \
        _parent_frame:=parallelopiped \
        _calib_file:=$(rospack find npm_calib)/config/parallelopiped_calib.yaml
"""
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from npm_calib import object_calib_lib as ocl


def se3_to_tf(T, parent, child, stamp):
    q = ocl.rotmat_to_quat_wxyz(T[:3, :3])
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x, tf.transform.translation.y, \
        tf.transform.translation.z = (float(v) for v in T[:3, 3])
    tf.transform.rotation.w, tf.transform.rotation.x, \
        tf.transform.rotation.y, tf.transform.rotation.z = (float(v) for v in q)
    return tf


def main():
    rospy.init_node("object_extrinsic_node")
    calib_file = rospy.get_param("~calib_file")
    parent = rospy.get_param("~parent_frame", ocl.DEFAULT_PARENT)
    child = rospy.get_param("~child_frame", ocl.DEFAULT_CHILD)

    T, meta = ocl.load_calib(calib_file)

    # The frames come from params but the rotation comes from the file, so
    # aiming one object's launch args at another object's calibration would
    # otherwise publish the wrong extrinsic under the right frame name.
    mismatch = ocl.frame_mismatch(meta, parent, child)
    # Identity is a legal transform, so an unusable calibration would otherwise
    # look exactly like a good one and silently mis-place every push point.
    if mismatch:
        rospy.logerr("object_extrinsic: %s was solved for different frames (%s)"
                     " - publishing IDENTITY %s -> %s. Calibrate this object, or"
                     " fix the `object` launch arg.", calib_file, mismatch,
                     parent, child)
        T = np.eye(4)
    elif not meta.get("calibrated"):
        rospy.logwarn("object_extrinsic: %s is NOT calibrated (%s) - publishing "
                      "IDENTITY %s -> %s. Run object_calib.launch first.",
                      calib_file,
                      "file missing" if meta.get("missing") else "calibrated: false",
                      parent, child)
        T = np.eye(4)
    else:
        rospy.loginfo("object_extrinsic: %s -> %s  %s", parent, child,
                      ocl.format_pose(T))

    caster = tf2_ros.StaticTransformBroadcaster()
    caster.sendTransform(se3_to_tf(T, parent, child, rospy.Time.now()))
    rospy.spin()


if __name__ == "__main__":
    main()

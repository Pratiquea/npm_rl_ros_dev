#!/usr/bin/env python3
"""
Publish the filtered object state in `world` continuously at mocap rate, on
/npm/object_state and /npm/object_odom. The single state source in the system;
every other node subscribes.

Example:
    rosrun npm_control object_state_node.py \
        _object_pose_topic:=/mocap_node/parallelopiped/pose \
        _object_parent_frame:=parallelopiped
"""
import rospy

from npm_control.object_state import ObjectStateEstimator


def main():
    rospy.init_node("object_state_node")
    ObjectStateEstimator()          # publishes on every accepted mocap sample
    rospy.spin()


if __name__ == "__main__":
    main()

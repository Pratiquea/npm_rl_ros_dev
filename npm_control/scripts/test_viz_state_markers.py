#!/usr/bin/env python3
"""
Rviz arrows for /npm/object_state: object velocity and direction to the goal.
Visualization only - it subscribes the published state, it does not filter.

Example:
    rosrun npm_control test_viz_state_markers.py
"""
import rospy
from visualization_msgs.msg import MarkerArray

from npm_control.object_state import ObjectStateClient
from npm_control import state_viz


class StateMarkers(object):
    def __init__(self):
        rate_hz = float(rospy.get_param("~marker_rate_hz", 20.0))
        self.vel_scale = float(rospy.get_param("~vel_arrow_scale", 1.0))
        self.client = ObjectStateClient()
        self.pub = rospy.Publisher("/npm/debug/state_markers", MarkerArray,
                                   queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate_hz), self._tick)

    def _tick(self, _evt):
        # Stale state publishes nothing, so a dead estimator shows as arrows
        # disappearing rather than a frozen arrow that still looks live.
        state = self.client.latest()
        if state is None:
            return
        self.pub.publish(state_viz.state_markers(state, vel_scale=self.vel_scale))


def main():
    rospy.init_node("test_viz_state_markers")
    StateMarkers()
    rospy.spin()


if __name__ == "__main__":
    main()

cmake_minimum_required(VERSION 3.16)
project(npm_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  visualization_msgs
  npm_msgs
  actionlib
  actionlib_msgs
  tf2_ros
  tf2_geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs visualization_msgs npm_msgs
                 actionlib actionlib_msgs tf2_ros tf2_geometry_msgs
)

catkin_install_python(PROGRAMS
  scripts/executor_node.py
  scripts/test_push_publisher.py
  scripts/test_push_action_static.py
  scripts/test_push_action_track.py
  scripts/coordinator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

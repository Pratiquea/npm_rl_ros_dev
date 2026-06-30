cmake_minimum_required(VERSION 3.16)
project(spot_calib)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/calibrate_from_bag.py
  scripts/data_collection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

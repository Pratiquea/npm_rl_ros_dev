cmake_minimum_required(VERSION 3.16)
project(npm_policy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  tf2_ros
  npm_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs visualization_msgs
                 tf2_ros npm_msgs
)

catkin_install_python(PROGRAMS
  scripts/policy_node.py
  scripts/test_viz_obs_cloud.py
  scripts/pcl_resample_viz_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

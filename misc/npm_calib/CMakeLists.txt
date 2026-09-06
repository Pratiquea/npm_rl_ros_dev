cmake_minimum_required(VERSION 3.16)
project(npm_calib)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  geometry_msgs
  sensor_msgs
  std_srvs
  visualization_msgs
  interactive_markers
  tf2_ros
  dynamic_reconfigure
  pluginlib
  rviz
)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

# rviz plugins are loaded by an already-running Qt app, so the moc output has to
# be generated for the Q_OBJECT panel; AUTOMOC does it without a wrap_cpp list.
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/ObjectExtrinsic.cfg
)

catkin_package(
  LIBRARIES npm_calib_panel
  CATKIN_DEPENDS rospy roscpp geometry_msgs sensor_msgs std_srvs
                 visualization_msgs interactive_markers tf2_ros
                 dynamic_reconfigure pluginlib rviz
)

include_directories(${catkin_INCLUDE_DIRS})

add_library(npm_calib_panel src/object_calib_panel.cpp)
target_link_libraries(npm_calib_panel ${catkin_LIBRARIES} Qt5::Widgets)
target_include_directories(npm_calib_panel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(TARGETS npm_calib_panel
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(FILES plugin_description.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/object_calib_node.py
  scripts/object_extrinsic_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()

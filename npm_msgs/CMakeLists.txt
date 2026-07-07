cmake_minimum_required(VERSION 3.16)
project(npm_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  actionlib_msgs
  message_generation
)

add_message_files(
  FILES
  PushCommand.msg
  PushResult.msg
  MacroState.msg
)

add_action_files(
  FILES
  ExecutePush.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs actionlib_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Filtered object state in `world` (goal at the origin). Produced by
# npm_control.object_state.ObjectStateEstimator from the mocap pose stream.
Header   header                    # stamp = mocap sample stamp, frame_id = world
geometry_msgs/Pose    pose         # object pose in world (ROS xyzw order)
geometry_msgs/Vector3 lin_vel      # m/s,   world, Kalman estimate
geometry_msgs/Vector3 ang_vel      # rad/s, world, Kalman estimate
float32  dist_to_goal              # m, horizontal distance from object to world origin
float32  dir_to_goal_x             # unit xy vector object -> goal
float32  dir_to_goal_y

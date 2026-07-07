# PushCommand.msg
# Two coordinate views of the same push:
# a) contact_point / push_force : world frame.
# b) push_point / force_body    : object (link) frame.
Header   header
geometry_msgs/Point   contact_point     # surface push point, world frame
geometry_msgs/Vector3 push_force        # 3d force vector, world frame, into surface
geometry_msgs/Point   push_point        # surface push point, object frame 
geometry_msgs/Vector3 force_body        # 3d force vector, object frame 
int32    loc_idx                        # original point index [0,128)

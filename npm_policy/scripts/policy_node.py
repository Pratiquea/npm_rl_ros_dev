#!/usr/bin/env python3
"""
Perform one SMDP macro-step of inference per /npm/infer call: build the 798-dim
observation from the settled Object state the coordinator provides, run the
policy to return a PushCommand.

Example:
    rosrun npm_policy policy_node.py _checkpoint_path:=<model.pt> _npz_path:=<obj.npz>
    rosrun npm_policy policy_node.py _use_oracle:=true _npz_path:=<obj.npz>
"""
import numpy as np
import rospy
import tf2_ros
import time
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from npm_msgs.msg import PushCommand, ObjectState
from npm_msgs.srv import InferPush, InferPushResponse

from npm_policy import policy_lib as pl
from npm_policy import obs_lib as ol
from npm_policy import obs_viz as ov
from npm_policy import oracle_lib as orc


class PolicyNode(object):
    def __init__(self):
        self.use_oracle = bool(rospy.get_param("~use_oracle", False))
        ckpt = rospy.get_param("~checkpoint_path", "") if self.use_oracle \
            else rospy.get_param("~checkpoint_path")
        npz = rospy.get_param("~npz_path")
        self.device = rospy.get_param("~device", "cpu")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.mass = float(rospy.get_param("~object_mass", 20.0))
        self.friction = float(rospy.get_param("~friction", 0.95))
        self.normal_len = float(rospy.get_param("~normal_marker_len", 0.05))
        self.raw_origin = list(rospy.get_param("~raw_obs_origin", [0.0, 0.0, 0.0]))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))
        # Contact-cloud margin (obs_lib.link_frame_cloud). Every node that
        # resolves loc_idx must run the same value; /npm/model_info carries the
        # effective scale so a disagreement is an error rather than a wrong push.
        self.shrink = float(rospy.get_param("~pcl_shrink", 1.0))

        # Reachability gate. Enforced here and nowhere else, so it covers the arm
        # executor and hand_push_server alike. Defaults mirror the sim
        # (objectmanip_env_discrete.py:144-145); floor_z is 0 because the world
        # origin is the mocap origin set by the ground-plane calibration.
        self.mask_reach = bool(rospy.get_param("~reach/enable", True))
        self.floor_z = float(rospy.get_param("~reach/floor_z", 0.0))
        self.ground_clearance = float(
            rospy.get_param("~reach/ground_clearance", pl.GROUND_CLEARANCE))
        self.normal_z_min = float(
            rospy.get_param("~reach/normal_z_min", pl.REACHABLE_NORMAL_Z_MIN))

        # Pointer sampling. top_k=1 is the deterministic argmax this node shipped
        # with; >1 draws uniformly from the k best REACHABLE points, so a contact
        # that cannot move the object is not re-picked forever off a frozen
        # observation. min_sep keeps the candidates from being neighbours on the
        # same face, which is what the top logits otherwise are.
        self.top_k = int(rospy.get_param("~infer/top_k", 1))
        self.min_sep = float(rospy.get_param("~infer/min_sep", 0.0))
        seed = int(rospy.get_param("~infer/seed", -1))
        self.rng = np.random.default_rng(None if seed < 0 else seed)
        # Last mask, for the debug draw. _act computes it; _publish_debug draws it.
        self.reach = None
        self.n_reach = ol.N_POINTS
        self.loc_unmasked = -1

        self.bundle = pl.PolicyBundle(ckpt, npz, self.mass, self.friction,
                                      device=self.device,
                                      load_net=not self.use_oracle,
                                      shrink=self.shrink)
        self.npz_path = npz
        inset_mm = 1000.0 * ol.shrink_inset(self.bundle.pts, self.bundle.scale,
                                            self.shrink)
        rospy.loginfo("policy_node: %s npz=%s (scale=%.4f pcl_shrink=%.4f -> "
                      "inset %.1f/%.1f/%.1f mm)",
                      "ORACLE (no checkpoint)" if self.use_oracle
                      else "ckpt=%s" % ckpt, npz, self.bundle.scale, self.shrink,
                      inset_mm[0], inset_mm[1], inset_mm[2])

        self.prev_action = np.zeros(4, dtype=np.float64)

        self.cmd_pub = rospy.Publisher("/npm/push_command", PushCommand, queue_size=1)
        self.prev_pub = rospy.Publisher("/npm/prev_action", Float32MultiArray,
                                        queue_size=1)
        # For Service transparency and bagging, policy node is provided all arguments and
        # every output is published for rosbag and rviz.
        self.obs_pub = rospy.Publisher("/npm/observation", Float32MultiArray,
                                       queue_size=1)
        self.state_pub = rospy.Publisher("/npm/served_object_state", ObjectState,
                                         queue_size=1)
        self.cloud_pub = rospy.Publisher("/npm/debug/obs_cloud", PointCloud2,
                                         queue_size=1)
        self.shrunk_pub = rospy.Publisher("/npm/debug/obs_cloud_shrunk", PointCloud2,
                                          queue_size=1)
        self.normals_pub = rospy.Publisher("/npm/debug/obs_normals", Marker,
                                           queue_size=1)
        self.raw_cloud_pub = rospy.Publisher("/npm/debug/raw_obs_cloud", PointCloud2,
                                             queue_size=1)
        self.raw_normals_pub = rospy.Publisher("/npm/debug/raw_obs_normals", Marker,
                                               queue_size=1)
        self.reach_pub = rospy.Publisher("/npm/debug/reachable", Marker,
                                         queue_size=1)
        self.tf_caster = tf2_ros.TransformBroadcaster()

        # Model info is published for verification if the npz (pcl) file is not the same as 
        # loaded by other nodes. this is done once at startup.
        # TODO: change it to better thing like service?
        self.model_pub = rospy.Publisher("/npm/model_info", String, queue_size=1,
                                         latch=True)
        self.model_pub.publish(
            String(ol.format_model_info(npz, self.bundle.scale, self.shrink)))

        self.srv = rospy.Service("/npm/infer", InferPush, self._on_infer)
        rospy.loginfo("policy_node: /npm/infer ready (device=%s, oracle=%s, "
                      "reach_mask=%s floor_z=%.3f clearance=%.3f normal_z_min=%.2f)",
                      self.device, self.use_oracle, self.mask_reach,
                      self.floor_z, self.ground_clearance, self.normal_z_min)

    def _on_infer(self, req):
        resp = InferPushResponse()
        if req.reset:
            self.prev_action = np.zeros(4, dtype=np.float64)

        state = req.state
        rot = ol.rotation_from_ros_quat(state.pose.orientation)
        pos = np.array([state.pose.position.x, state.pose.position.y,
                        state.pose.position.z])
        lin_vel = np.array([state.lin_vel.x, state.lin_vel.y, state.lin_vel.z])
        ang_vel = np.array([state.ang_vel.x, state.ang_vel.y, state.ang_vel.z])

        try:
            obs, raw, loc, force_body, force_world, contact_link, next_prev = \
                self._act(rot, pos, lin_vel, ang_vel)
        except Exception as e:
            rospy.logerr("policy_node: inference failed: %s", e)
            resp.success = False
            resp.message = str(e)
            return resp

        stamp = state.header.stamp if state.header.stamp != rospy.Time(0) \
            else rospy.Time.now()
        cmd = PushCommand()
        cmd.header.stamp = stamp
        cmd.header.frame_id = self.world_frame

        # loc_idx is the contact point. Policy node simply forwards it to the executor, which
        # resolves it against its the same npz model(pcl). push_point/contact_point are no 
        # longer used but kept for testing purposes.

        cmd.loc_idx = int(loc)
        cmd.force_body = Vector3(float(force_body[0]), float(force_body[1]),
                                 float(force_body[2]))
        cmd.push_force = Vector3(float(force_world[0]), float(force_world[1]),
                                 float(force_world[2]))
        cmd.push_point = Point(*[float(v) for v in contact_link])
        contact_world = np.asarray(rot.apply(contact_link)) + pos
        cmd.contact_point = Point(*[float(v) for v in contact_world])

        # The decoded action, not the net's pre-squash output: the sim writes
        # [loc_idx, |F|/F_max, u1, u2] back into the next observation.
        self.prev_action = next_prev
        resp.command = cmd
        resp.success = True
        self._publish_debug(obs, state, cmd, next_prev, rot, pos, stamp)
        # Report when the gate moved the choice: that is the livelock this exists
        # to prevent, and it is invisible otherwise.
        moved = "" if self.reach is None or self.loc_unmasked == loc \
            else " (mask moved it from %d)" % self.loc_unmasked
        reach = "" if self.reach is None \
            else " reach=%d/%d" % (self.n_reach, ol.N_POINTS)
        rospy.loginfo("infer[%s]: loc=%d z=%.3f |f|=%.1fN dist=%.3f%s%s",
                      "oracle" if self.use_oracle else "policy", loc,
                      float(contact_world[2]),
                      float(np.linalg.norm(force_world)), state.dist_to_goal,
                      reach, moved)
        return resp

    def _act(self, rot, pos, lin_vel, ang_vel):
        """One macro-step action.

        Returns (obs, raw, loc, force_body, force_world, contact_link, next_prev),
        where contact_link is the chosen surface point in the object link frame.

        The observation is assembled on both paths so /npm/observation and the
        debug clouds stay identical between oracle and policy runs, even though
        the oracle reads only the pose. Post-processing is shared, so both paths
        reproduce objectmanip_env_discrete.step(): decode the cone action about
        the body-frame normal at loc.

        rot/pos are used BOTH to assemble the observation and to build the cone
        frame, which is what the sim does (it decodes against the pose cached
        during the previous _get_observations). Re-reading a fresher pose for
        either half would decode the action in a frame the policy never saw.
        """
        b = self.bundle
        obs = ol.assemble_observation(rot, pos, lin_vel, ang_vel, self.prev_action,
                                      b.mass, b.friction, b.scale, b.pts, b.normals)
        self.reach, self.n_reach, self.loc_unmasked = None, ol.N_POINTS, -1
        if self.use_oracle:
            # The oracle is left alone. oracle_lib documents that the sim's mask
            # was deliberately not ported into it.
            raw = orc.oracle_raw_action(rot, pos, b.pts, b.normals, b.scale,
                                        b.centroid, shrink=b.shrink)
        else:
            rospy.loginfo("policy_node infer begin ")
            time_start = time.perf_counter()
            if self.mask_reach:
                # Same rot/pos as the observation, per this method's contract.
                self.reach, pts_world = pl.reachable_mask(
                    rot, pos, b.link_pts, b.normals, floor_z=self.floor_z,
                    ground_clearance=self.ground_clearance,
                    normal_z_min=self.normal_z_min, return_world=True)
                raw, self.n_reach, self.loc_unmasked = pl.infer_raw_action_masked(
                    b.net, b.mean, b.std, obs, self.reach, b.device,
                    top_k=self.top_k, rng=self.rng, min_sep=self.min_sep,
                    pts_world=pts_world)
                if self.n_reach == 0:
                    # Every score took the same penalty, so this is the unmasked
                    # choice. Nothing is pushable: the object is probably flipped,
                    # mis-calibrated, or the floor plane is wrong.
                    rospy.logwarn_throttle(
                        2.0, "policy_node: no reachable point on the object; "
                             "falling back to the unmasked argmax (loc=%d). "
                             "Check reach/floor_z=%.3f against the object pose.",
                        int(raw[0]), self.floor_z)
            elif self.top_k > 1:
                # Gate off but sampling on: every point is a candidate. self.reach
                # stays None so the debug marker keeps meaning "gate enforced".
                raw, _n, self.loc_unmasked = pl.infer_raw_action_masked(
                    b.net, b.mean, b.std, obs,
                    np.ones(ol.N_POINTS, dtype=bool), b.device,
                    top_k=self.top_k, rng=self.rng, min_sep=self.min_sep,
                    pts_world=np.asarray(rot.apply(b.link_pts)) + pos)
            else:
                raw = pl.infer_raw_action(b.net, b.mean, b.std, obs, b.device)
            elapsed_ms = (time.perf_counter() - time_start)*1000.0
            rospy.loginfo("policy_node infer complete, time taken (ms) = {:.2f}".format(elapsed_ms))
        loc, force_body, contact_link, _n, next_prev = pl.postprocess(
            raw, rot, pos, b.pts, b.normals, b.scale, b.centroid, b.shrink)
        force_world = np.asarray(rot.apply(force_body), dtype=np.float64)
        return obs, raw, loc, force_body, force_world, contact_link, next_prev

    def _publish_debug(self, obs, state, cmd, next_prev, rot, pos, stamp):
        self.cmd_pub.publish(cmd)
        pa = Float32MultiArray()
        # Decoded [loc_idx, |F|/F_max, u1, u2]. u1/u2 read as lateral/tilt only
        # while the cone frame is anchored to world up; on near-top and
        # near-bottom faces the anchor blends to the goal direction and u2 means
        # "tilt toward the goal" instead.
        pa.data = [float(v) for v in next_prev]
        self.prev_pub.publish(pa)
        if not self.publish_debug:
            return

        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="obs", size=ol.OBS_LEN,
                                              stride=ol.OBS_LEN)]
        msg.data = obs.tolist()
        self.obs_pub.publish(msg)
        self.state_pub.publish(state)

        # Draw the LINK-frame cloud, not obs[PC_OFF:NORMALS_OFF]: the obs block is
        # unscaled and COM-centered, so it would neither overlay the object nor
        # match what test_viz_obs_cloud puts on these same topics. The normals
        # block is already the rotated normals, which need no offset.
        # obs_cloud is the NOMINAL model; obs_cloud_shrunk is the cloud this node
        # actually resolved loc_idx against. Normals start on the nominal surface.
        nominal_rotated = ol.rotate_cloud(rot, self.bundle.link_pts_nominal)
        link_rotated = ol.rotate_cloud(rot, self.bundle.link_pts)
        tf, cloud, normals = ov.cloud_and_normals(
            nominal_rotated, obs[ol.NORMALS_OFF:ol.EXTRAS_OFF],
            pos, stamp, parent=self.world_frame, length=self.normal_len)
        self.tf_caster.sendTransform(tf)
        self.cloud_pub.publish(cloud)
        self.normals_pub.publish(normals)
        self.shrunk_pub.publish(ov.cloud_msg(link_rotated, stamp))

        # The gate is computed on the SHRUNK points, so the marker is drawn there
        # too: it must colour the points the argmax actually chose between.
        if self.reach is not None:
            self.reach_pub.publish(
                ov.reach_marker(link_rotated, self.reach, stamp))

        # The blocks as the net saw them: unscaled and COM-centered, so they are
        # parked at the goal instead of on the object.
        tf, cloud, normals = ov.raw_obs_cloud_and_normals(
            obs, self.raw_origin, stamp, parent=self.world_frame,
            length=self.normal_len)
        self.tf_caster.sendTransform(tf)
        self.raw_cloud_pub.publish(cloud)
        self.raw_normals_pub.publish(normals)


def main():
    rospy.init_node("policy_node")
    PolicyNode()
    rospy.spin()


if __name__ == "__main__":
    main()

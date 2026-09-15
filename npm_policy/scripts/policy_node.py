#!/usr/bin/env python3
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
        self.shrink = float(rospy.get_param("~pcl_shrink", 1.0))

        self.mask_reach = bool(rospy.get_param("~reach/enable", True))
        self.floor_z = float(rospy.get_param("~reach/floor_z", 0.0))
        self.ground_clearance = float(
            rospy.get_param("~reach/ground_clearance", pl.GROUND_CLEARANCE))
        self.normal_z_min = float(
            rospy.get_param("~reach/normal_z_min", pl.REACHABLE_NORMAL_Z_MIN))

        self.top_k = int(rospy.get_param("~infer/top_k", 1))
        self.min_sep = float(rospy.get_param("~infer/min_sep", 0.0))
        seed = int(rospy.get_param("~infer/seed", -1))
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.taper = bool(rospy.get_param("~taper/enable", False))
        self.taper_dist = float(rospy.get_param("~taper/taper_dist", 1.0))
        self.taper_force_min = float(rospy.get_param("~taper/taper_force_min", 25.0))
        self.taper_p_far = float(rospy.get_param("~taper/p_far", 0.1))
        self.taper_p_near = float(rospy.get_param("~taper/p_near", 0.8))
        self.taper_prev = bool(rospy.get_param("~taper/write_prev_action", True))
        self.taper_rng = np.random.default_rng(None if seed < 0 else seed + 1)
        if self.taper and not 0.0 <= self.taper_force_min <= pl.FORCE_MAX:
            raise ValueError(
                "taper/taper_force_min=%.1f N is outside [0, %.1f]"
                % (self.taper_force_min, pl.FORCE_MAX))
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


        cmd.loc_idx = int(loc)
        cmd.force_body = Vector3(float(force_body[0]), float(force_body[1]),
                                 float(force_body[2]))
        cmd.push_force = Vector3(float(force_world[0]), float(force_world[1]),
                                 float(force_world[2]))
        cmd.push_point = Point(*[float(v) for v in contact_link])
        contact_world = np.asarray(rot.apply(contact_link)) + pos
        cmd.contact_point = Point(*[float(v) for v in contact_world])

        self.prev_action = next_prev
        resp.command = cmd
        resp.success = True
        self._publish_debug(obs, state, cmd, next_prev, rot, pos, stamp)
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
        b = self.bundle
        obs = ol.assemble_observation(rot, pos, lin_vel, ang_vel, self.prev_action,
                                      b.mass, b.friction, b.scale, b.pts, b.normals)
        self.reach, self.n_reach, self.loc_unmasked = None, ol.N_POINTS, -1
        if self.use_oracle:
            raw = orc.oracle_raw_action(rot, pos, b.pts, b.normals, b.scale,
                                        b.centroid, shrink=b.shrink)
        else:
            rospy.loginfo("policy_node infer begin ")
            time_start = time.perf_counter()
            if self.mask_reach:
                self.reach, pts_world = pl.reachable_mask(
                    rot, pos, b.link_pts, b.normals, floor_z=self.floor_z,
                    ground_clearance=self.ground_clearance,
                    normal_z_min=self.normal_z_min, return_world=True)
                raw, self.n_reach, self.loc_unmasked = pl.infer_raw_action_masked(
                    b.net, b.mean, b.std, obs, self.reach, b.device,
                    top_k=self.top_k, rng=self.rng, min_sep=self.min_sep,
                    pts_world=pts_world)
                if self.n_reach == 0:
                    rospy.logwarn_throttle(
                        2.0, "policy_node: no reachable point on the object; "
                             "falling back to the unmasked argmax (loc=%d). "
                             "Check reach/floor_z=%.3f against the object pose.",
                        int(raw[0]), self.floor_z)
            elif self.top_k > 1:
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
        if self.taper:
            force_body = pl.apply_force_taper(
                force_body, float(np.linalg.norm(pos[:2])), self.taper_dist,
                self.taper_force_min, self.taper_p_far, self.taper_p_near,
                self.taper_rng)
            if self.taper_prev:
                next_prev[1] = float(np.linalg.norm(force_body)) / pl.FORCE_MAX
        force_world = np.asarray(rot.apply(force_body), dtype=np.float64)
        return obs, raw, loc, force_body, force_world, contact_link, next_prev

    def _publish_debug(self, obs, state, cmd, next_prev, rot, pos, stamp):
        self.cmd_pub.publish(cmd)
        pa = Float32MultiArray()
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

        nominal_rotated = ol.rotate_cloud(rot, self.bundle.link_pts_nominal)
        link_rotated = ol.rotate_cloud(rot, self.bundle.link_pts)
        tf, cloud, normals = ov.cloud_and_normals(
            nominal_rotated, obs[ol.NORMALS_OFF:ol.EXTRAS_OFF],
            pos, stamp, parent=self.world_frame, length=self.normal_len)
        self.tf_caster.sendTransform(tf)
        self.cloud_pub.publish(cloud)
        self.normals_pub.publish(normals)
        self.shrunk_pub.publish(ov.cloud_msg(link_rotated, stamp))

        if self.reach is not None:
            self.reach_pub.publish(
                ov.reach_marker(link_rotated, self.reach, stamp))

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

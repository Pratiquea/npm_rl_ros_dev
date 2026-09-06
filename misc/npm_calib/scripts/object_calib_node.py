#!/usr/bin/env python3
"""
Interactive tuner for the <mocap body> -> object_link extrinsic: drag the model
cloud onto the physical object in rviz, or type exact values in
rqt_reconfigure, then save.

Example:
    roslaunch npm_calib object_calib.launch object:=parallelopiped
    rosrun npm_calib object_calib_node.py _npz_path:=<obj.npz>
"""
import numpy as np
import rospy
import tf2_ros
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Pose, TransformStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                    Marker)

from npm_calib import object_calib_lib as ocl
from npm_calib.cfg import ObjectExtrinsicConfig
from npm_policy import obs_lib as ol
from npm_policy import obs_viz as ov

CFG_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
# Below this the six reconfigure fields are considered unchanged, so the echo
# guard is never armed for an update that would not fire a callback anyway.
CFG_EPS = 1e-9


def se3_to_tf(T, parent, child, stamp):
    q = ocl.rotmat_to_quat_wxyz(T[:3, :3])
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x, tf.transform.translation.y, \
        tf.transform.translation.z = (float(v) for v in T[:3, 3])
    tf.transform.rotation.w, tf.transform.rotation.x, \
        tf.transform.rotation.y, tf.transform.rotation.z = (float(v) for v in q)
    return tf


def se3_to_pose(T):
    q = ocl.rotmat_to_quat_wxyz(T[:3, :3])
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = \
        (float(v) for v in q)
    return p


def pose_to_se3(pose):
    R = ocl.quat_wxyz_to_rotmat([pose.orientation.w, pose.orientation.x,
                                 pose.orientation.y, pose.orientation.z])
    return ocl.make_se3(R, [pose.position.x, pose.position.y, pose.position.z])


def _axis_controls():
    """Six move/rotate handles on the mocap body axes.

    orientation_mode INHERIT makes the handles turn with the current estimate,
    so dragging stays intuitive after a large rotation.
    """
    controls = []
    axes = (("x", (1.0, 0.0, 0.0)), ("y", (0.0, 0.0, 1.0)), ("z", (0.0, 1.0, 0.0)))
    for name, (qx, qy, qz) in axes:
        for mode, suffix in ((InteractiveMarkerControl.MOVE_AXIS, "move"),
                             (InteractiveMarkerControl.ROTATE_AXIS, "rotate")):
            c = InteractiveMarkerControl()
            c.name = "%s_%s" % (suffix, name)
            c.orientation.w = 1.0
            c.orientation.x, c.orientation.y, c.orientation.z = qx, qy, qz
            norm = np.linalg.norm([c.orientation.w, c.orientation.x,
                                   c.orientation.y, c.orientation.z])
            c.orientation.w /= norm
            c.orientation.x /= norm
            c.orientation.y /= norm
            c.orientation.z /= norm
            c.interaction_mode = mode
            c.orientation_mode = InteractiveMarkerControl.INHERIT
            controls.append(c)
    return controls


class ObjectCalibNode(object):

    def __init__(self):
        npz = rospy.get_param("~npz_path")
        self.calib_file = rospy.get_param("~calib_file")
        self.parent_frame = rospy.get_param("~parent_frame", ocl.DEFAULT_PARENT)
        self.child_frame = rospy.get_param("~child_frame", ocl.DEFAULT_CHILD)
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.marker_scale = float(rospy.get_param("~marker_scale", 0.5))
        self.normal_len = float(rospy.get_param("~normal_marker_len", 0.05))
        self.publish_rate = float(rospy.get_param("~publish_rate_hz", 20.0))
        self.readout_rate = float(rospy.get_param("~readout_rate_hz", 2.0))

        self.npz_path = npz
        # Alignment is done against the TRUE silhouette, so the tuner keeps
        # shrink=1.0 here. The contracted cloud the robot actually aims at is
        # published beside it on ~cloud_shrunk, in its own rviz colour.
        self.shrink = float(rospy.get_param("~pcl_shrink", 1.0))
        pts, self.normals, self.scale, centroid = ol.load_model_npz(npz)
        self.link_pts = ol.link_frame_cloud(pts, self.scale, centroid)
        self.link_pts_shrunk = ol.link_frame_cloud(pts, self.scale, centroid,
                                                   self.shrink)
        self.inset_mm = 1000.0 * ol.shrink_inset(pts, self.scale, self.shrink)

        self.T, meta = ocl.load_calib(self.calib_file)
        # Seeding from a file solved for another object would hand the operator a
        # plausible-looking but wrong starting pose, which is harder to spot than
        # starting from identity.
        mismatch = ocl.frame_mismatch(meta, self.parent_frame, self.child_frame)
        if mismatch:
            rospy.logwarn("object_calib: %s was solved for different frames (%s)"
                          " - seeding from IDENTITY instead.", self.calib_file,
                          mismatch)
            self.T = np.eye(4)
        # The extrinsic is carried as R_coarse @ R_residual: R_coarse is a
        # cube-group element, and only the residual is exposed as roll/pitch/yaw.
        # See object_calib_lib.residual_rpy_from_se3 for why absolute RPY cannot
        # be the edit handle.
        self.R_coarse = ocl.nearest_cube_rotation(self.T[:3, :3])
        rospy.loginfo("object_calib: seed %s (calibrated=%s) %s",
                      self.calib_file, meta.get("calibrated"),
                      ocl.format_pose(self.T, self.R_coarse))

        # One-shot flag cleared at the top of the reconfigure callback: without
        # it a drag would echo back through update_configuration and fight the
        # very drag that produced it.
        self._echo = False
        self._seeding = True
        self._last_cfg = None
        # The marker server is up before the reconfigure server, so early rviz
        # feedback must find something to test against.
        self.dyn = None

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        self.tf_caster = tf2_ros.TransformBroadcaster()

        # Latched so a late subscriber sees the cloud at once, but also
        # republished every tick: the points never move in the child frame, yet
        # rviz drops a cloud whose stamp has fallen out of the tf buffer, so a
        # single startup publish disappears from the display after ~10 s.
        self.cloud_pub = rospy.Publisher("~cloud", PointCloud2, queue_size=1,
                                         latch=True)
        self.shrunk_pub = rospy.Publisher("~cloud_shrunk", PointCloud2, queue_size=1,
                                          latch=True)
        self.normals_pub = rospy.Publisher("~normals", Marker, queue_size=1)
        self.readout_pub = rospy.Publisher("~readout", Marker, queue_size=1)
        stamp = rospy.Time.now()
        self.cloud_pub.publish(ov.cloud_msg(self.link_pts.reshape(-1), stamp,
                                            frame=self.child_frame))
        self.shrunk_pub.publish(ov.cloud_msg(self.link_pts_shrunk.reshape(-1), stamp,
                                             frame=self.child_frame))

        self.im_server = InteractiveMarkerServer("~calib")
        self.menu = MenuHandler()
        self._build_menu()
        self._build_marker()

        self.dyn = Server(ObjectExtrinsicConfig, self._on_reconfigure)
        self._seeding = False
        self._push_to_reconfigure()

        rospy.Service("~save", Trigger, self._srv_save)
        rospy.Service("~reset", Trigger, self._srv_reset)
        rospy.Service("~reset_saved", Trigger, self._srv_reset_saved)
        rospy.Service("~snap90", Trigger, self._srv_snap90)
        rospy.Service("~zero_translation", Trigger, self._srv_zero_translation)
        rospy.Service("~zero_rotation", Trigger, self._srv_zero_rotation)

        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self._tick)
        rospy.Timer(rospy.Duration(1.0 / self.readout_rate), self._readout)
        rospy.loginfo("object_calib: %s -> %s, %d pts (scale=%.4f); ~cloud is the "
                      "nominal model, ~cloud_shrunk is pcl_shrink=%.4f "
                      "(inset %.1f/%.1f/%.1f mm)",
                      self.parent_frame, self.child_frame, len(self.link_pts),
                      self.scale, self.shrink, self.inset_mm[0], self.inset_mm[1],
                      self.inset_mm[2])


    def _build_menu(self):
        self.menu.insert("Save calibration", callback=self._menu_save)
        self.menu.insert("Snap rotation to nearest 90 deg",
                         callback=self._menu_snap90)
        reset = self.menu.insert("Reset")
        self.menu.insert("To identity", parent=reset, callback=self._menu_reset)
        self.menu.insert("To last saved", parent=reset,
                         callback=self._menu_reset_saved)
        self.menu.insert("Zero translation", parent=reset,
                         callback=self._menu_zero_translation)
        self.menu.insert("Zero rotation", parent=reset,
                         callback=self._menu_zero_rotation)

    def _build_marker(self):
        im = InteractiveMarker()
        # Anchored in the mocap body frame, so the handle pose IS the extrinsic
        # being solved for, and it rides the object when it is hand-moved.
        im.header.frame_id = self.parent_frame
        im.name = "object_extrinsic"
        im.description = "%s -> %s" % (self.parent_frame, self.child_frame)
        im.scale = self.marker_scale
        im.pose = se3_to_pose(self.T)

        handle = Marker()
        handle.type = Marker.SPHERE
        handle.scale.x = handle.scale.y = handle.scale.z = 0.1 * self.marker_scale
        handle.color = ColorRGBA(1.0, 0.6, 0.0, 0.9)

        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.MENU
        menu_control.always_visible = True
        menu_control.name = "menu"
        menu_control.markers.append(handle)
        im.controls.append(menu_control)
        im.controls.extend(_axis_controls())

        self.im_server.insert(im, self._on_marker_feedback)
        self.menu.apply(self.im_server, im.name)
        self.im_server.applyChanges()


    def _on_marker_feedback(self, feedback):
        if feedback.event_type != feedback.POSE_UPDATE:
            return
        self.T = pose_to_se3(feedback.pose)
        self._push_to_reconfigure()

    def _on_reconfigure(self, config, level):
        if self._seeding:
            return config
        # Cleared here rather than after update_configuration returns:
        # dynamic_reconfigure dispatches this callback synchronously on some
        # paths and asynchronously on others, and only clear-inside is correct
        # under both.
        if self._echo:
            self._echo = False
            self._last_cfg = [config[f] for f in CFG_FIELDS]
            return config

        self.T = ocl.se3_from_coarse_and_residual(
            self.R_coarse, [config.x, config.y, config.z],
            [config.roll, config.pitch, config.yaw])
        self._last_cfg = [config[f] for f in CFG_FIELDS]
        self._push_to_marker()
        return config

    def _push_to_reconfigure(self):
        # setPose does not re-enter the feedback callback, so only the
        # reconfigure direction needs the echo guard.
        if self.dyn is None:
            return
        self._reseat_coarse()
        xyz, rpy = ocl.residual_rpy_from_se3(self.T, self.R_coarse)
        values = [float(v) for v in list(xyz) + list(rpy)]
        if self._last_cfg is not None and \
                all(abs(a - b) < CFG_EPS for a, b in zip(values, self._last_cfg)):
            return                                  # no callback would fire
        self._last_cfg = values
        self._echo = True
        self.dyn.update_configuration(dict(zip(CFG_FIELDS, values)))

    def _reseat_coarse(self, force=False):
        """Re-centre the residual on the cube element the pose has drifted to.

        Guarded by RESEAT_ANGLE_DEG so a drag near a Voronoi boundary cannot
        flip the coarse element back and forth: that would swap the pitch=+-90
        discontinuity for a boundary one instead of removing it. force is for
        deliberate jumps (reset, snap, load), where there is no drag to disturb.
        """
        nearest = ocl.nearest_cube_rotation(self.T[:3, :3])
        if np.allclose(nearest, self.R_coarse):
            return False
        if not force and \
                ocl.residual_angle_deg(self.T, self.R_coarse) <= ocl.RESEAT_ANGLE_DEG:
            return False
        self.R_coarse = nearest
        rospy.loginfo("object_calib: coarse re-seated to [%s]",
                      ocl.format_cube_rotation(self.R_coarse))
        return True

    def _push_to_marker(self):
        self.im_server.setPose("object_extrinsic", se3_to_pose(self.T))
        self.im_server.applyChanges()

    def _set(self, T, what):
        self.T = np.asarray(T, dtype=np.float64)
        self._reseat_coarse(force=True)
        self._push_to_reconfigure()
        self._push_to_marker()
        rospy.loginfo("object_calib: %s -> %s", what,
                      ocl.format_pose(self.T, self.R_coarse).replace("\n", " | "))


    def _reset(self):
        self._set(np.eye(4), "reset to identity")

    def _reset_saved(self):
        T, meta = ocl.load_calib(self.calib_file)
        mismatch = ocl.frame_mismatch(meta, self.parent_frame, self.child_frame)
        if mismatch:
            rospy.logwarn("object_calib: refusing to reset to %s (%s)",
                          self.calib_file, mismatch)
            return
        self._set(T, "reset to saved (calibrated=%s)" % meta.get("calibrated"))

    def _snap90(self):
        # Equivalent to zeroing roll/pitch/yaw: _set re-seats the coarse element
        # to the snapped rotation, which leaves an exactly identity residual.
        self._set(ocl.snap_rotation_to_90(self.T), "snapped rotation")

    def _zero_translation(self):
        self._set(ocl.make_se3(self.T[:3, :3], np.zeros(3)), "zeroed translation")

    def _zero_rotation(self):
        self._set(ocl.make_se3(np.eye(3), self.T[:3, 3]), "zeroed rotation")

    def _save(self):
        res = self._residual()
        ocl.save_calib(self.calib_file, self.T, parent=self.parent_frame,
                       child=self.child_frame, calibrated=True,
                       note="manual rviz alignment", npz_path=self.npz_path,
                       floor_residual=res)
        msg = "saved %s: %s" % (self.calib_file,
                                ocl.format_pose(self.T, self.R_coarse).replace("\n", " | "))
        if res is not None:
            msg += "  (min_z=%.4f m tilt=%.2f deg)" % (res["min_z"],
                                                       res["tilt_deg"])
        rospy.loginfo("object_calib: %s", msg)
        return msg


    def _world_T_parent(self):
        try:
            tf = self.tf_buf.lookup_transform(self.world_frame, self.parent_frame,
                                              rospy.Time(0), rospy.Duration(0.1))
        except tf2_ros.TransformException as exc:
            rospy.logwarn_throttle(5.0, "object_calib: TF %s <- %s: %s",
                                   self.world_frame, self.parent_frame, exc)
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return ocl.make_se3(ocl.quat_wxyz_to_rotmat([q.w, q.x, q.y, q.z]),
                            [t.x, t.y, t.z])

    def _residual(self):
        T_world_parent = self._world_T_parent()
        if T_world_parent is None:
            return None
        res = ocl.floor_residual(self.link_pts, T_world_parent, self.T)
        return {k: (float(v) if isinstance(v, float) else v)
                for k, v in res.items()}

    def _tick(self, _event):
        stamp = rospy.Time.now()
        self.tf_caster.sendTransform(
            se3_to_tf(self.T, self.parent_frame, self.child_frame, stamp))
        self.cloud_pub.publish(ov.cloud_msg(self.link_pts.reshape(-1), stamp,
                                            frame=self.child_frame))
        self.shrunk_pub.publish(ov.cloud_msg(self.link_pts_shrunk.reshape(-1), stamp,
                                             frame=self.child_frame))
        self.normals_pub.publish(
            ov.normals_marker(self.link_pts.reshape(-1),
                              np.asarray(self.normals).reshape(-1), stamp,
                              frame=self.child_frame, length=self.normal_len))

    def _readout(self, _event):
        res = self._residual()
        text = ocl.format_pose(self.T, self.R_coarse)
        if res is None:
            text += "\nno %s TF: floor residual unavailable" % self.parent_frame
        elif res["flat"]:
            text += "\nmin_z %+.4f m  tilt %.2f deg  sunk %d" % (
                res["min_z"], res["tilt_deg"], res["n_below"])
        else:
            # min_z only means something while the object rests flat; reporting
            # it for a tilted object would be tuning against noise.
            text += "\ntilt %.2f deg (not flat: min_z not meaningful)" % \
                res["tilt_deg"]
        rospy.loginfo_throttle(5.0, "object_calib: %s", text.replace("\n", " | "))

        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.child_frame
        m.ns = "calib_readout"
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.z = 1.2
        m.pose.orientation.w = 1.0
        m.scale.z = 0.08
        m.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        m.text = text
        m.lifetime = rospy.Duration(2.0 / max(self.readout_rate, 1e-3))
        self.readout_pub.publish(m)


    def _menu_save(self, _fb):
        self._save()

    def _menu_reset(self, _fb):
        self._reset()

    def _menu_reset_saved(self, _fb):
        self._reset_saved()

    def _menu_snap90(self, _fb):
        self._snap90()

    def _menu_zero_translation(self, _fb):
        self._zero_translation()

    def _menu_zero_rotation(self, _fb):
        self._zero_rotation()

    def _srv_save(self, _req):
        return TriggerResponse(success=True, message=self._save())

    def _srv_reset(self, _req):
        self._reset()
        return TriggerResponse(success=True, message=ocl.format_pose(self.T, self.R_coarse))

    def _srv_reset_saved(self, _req):
        self._reset_saved()
        return TriggerResponse(success=True, message=ocl.format_pose(self.T, self.R_coarse))

    def _srv_snap90(self, _req):
        self._snap90()
        return TriggerResponse(success=True, message=ocl.format_pose(self.T, self.R_coarse))

    def _srv_zero_translation(self, _req):
        self._zero_translation()
        return TriggerResponse(success=True, message=ocl.format_pose(self.T, self.R_coarse))

    def _srv_zero_rotation(self, _req):
        self._zero_rotation()
        return TriggerResponse(success=True, message=ocl.format_pose(self.T, self.R_coarse))


def main():
    rospy.init_node("object_calib_node")
    ObjectCalibNode()
    rospy.spin()


if __name__ == "__main__":
    main()

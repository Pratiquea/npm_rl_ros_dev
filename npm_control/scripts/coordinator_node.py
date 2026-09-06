#!/usr/bin/env python3
"""
SMDP macro-step state machine: IDLE -> INFER -> PUSH -> SETTLE -> (success? stop
: INFER). One macro-step = one inference + one push + one settle.

Object state is subscribed from object_state_node (/npm/object_state), not
filtered here. INFER calls the policy's /npm/infer service with that settled
state, so exactly one inference happens per macro-step and the policy sees the
same sample this node judged settled. PUSH delegates to the executor's
ExecutePush action and is preempted if the object enters the success circle
mid-push.

Example:
    rosrun npm_control coordinator_node.py
"""
import sys
import threading

import numpy as np
import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from npm_msgs.msg import MacroState, ExecutePushAction, ExecutePushGoal
from npm_msgs.srv import InferPush, InferPushRequest

from npm_control.object_state import ObjectStateClient
from npm_control import executor_lib as el

IDLE, INFER, PUSH, SETTLE = 0, 1, 2, 3
NAMES = {IDLE: "IDLE", INFER: "INFER", PUSH: "PUSH", SETTLE: "SETTLE"}
_STATUS_NAMES = {getattr(GoalStatus, n): n for n in
                 ("PENDING", "ACTIVE", "PREEMPTED", "SUCCEEDED", "ABORTED",
                  "REJECTED", "PREEMPTING", "RECALLING", "RECALLED", "LOST")}


class _GateSetupError(RuntimeError):
    pass


class _Gate(object):
    """One pending operator prompt, answered on a worker thread.

    _tick runs on a rospy Timer, so blocking it on stdin would freeze
    /npm/macro_state and the success-circle preemption for as long as the
    operator takes to answer. The prompt therefore runs on its own thread and
    _tick polls reply(), which is None until an answer lands.
    """

    def __init__(self, banner, allow_skip):
        self.allow_skip = allow_skip
        self._reply = None
        self._lock = threading.Lock()
        print(banner)
        sys.stdout.flush()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            try:
                answer = input().strip().lower()
            except (EOFError, OSError):
                # Fail closed. No stdin means nobody is watching, and an
                # unattended push is what this gate exists to prevent. Unlike
                # executor_node._gate, which only paces debugging and may
                # auto-continue, this one ends the episode. The startup isatty()
                # check should have caught it before anything moved.
                rospy.logerr("confirm gate: stdin closed, ending the episode.")
                answer = "q"
            if answer in ("", "q") or (answer == "s" and self.allow_skip):
                break
            # An unrecognized key must never fall through to "continue".
            print("   unrecognized key %r, try again." % answer)
            sys.stdout.flush()
        with self._lock:
            self._reply = answer

    def reply(self):
        with self._lock:
            return self._reply


class Coordinator(object):
    # Own namespace so the not-yet-approved push does not collide with the
    # executor's committed markers, which share the topic under "npm_push".
    PREVIEW_NS = "npm_confirm"

    def __init__(self):
        # Real-robot operator gates, armed by npm_stack.launch from confirm:=true.
        # Checked before any other setup: with no stdin the gates cannot ask, and
        # a gate that cannot ask is not a gate, so refuse to run at all rather
        # than silently pushing unattended.
        self.confirm_settle = bool(rospy.get_param("~confirm/settle", False))
        self.confirm_pre_push = bool(rospy.get_param("~confirm/pre_push", False))
        self.gate_armed = self.confirm_settle or self.confirm_pre_push
        if self.gate_armed and not sys.stdin.isatty():
            rospy.logfatal(
                "confirm gates are armed but this node has no stdin. roslaunch "
                "gives nodes no tty: launch with confirm:=true so npm_stack.launch "
                "starts the coordinator under 'xterm -e', or rosrun it from a "
                "shell. Refusing to start.")
            rospy.signal_shutdown("confirm gates armed without a tty")
            raise _GateSetupError("confirm gates armed without a tty")

        self.push_dwell = float(rospy.get_param("~push_dwell_s", 2.0))
        self.settle_vel = float(rospy.get_param("~settle_vel_thr", 0.1))
        self.settle_ang = float(rospy.get_param("~settle_ang_vel_thr", 0.1))
        self.settle_timeout = float(rospy.get_param("~settle_timeout_s", 10.0))
        # A single settled sample is not a settled object: one velocity dip while
        # a hand (or the arm) is still on it would start the next macro-step.
        self.settle_hold = float(rospy.get_param("~settle_hold_s", 0.5))
        # 0 = unlimited. Without a cap the loop runs forever on an object that
        # never reaches the circle.
        self.max_steps = int(rospy.get_param("~max_macro_steps", 0))
        self.success_radius = float(rospy.get_param("~success_radius", 0.2))
        self.infer_timeout = float(rospy.get_param("~infer_timeout_s", 5.0))
        rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.arrow_len = float(rospy.get_param("~viz/arrow_len", 0.3))

        self.est = ObjectStateClient()
        self.cmd = None
        self.phase = IDLE
        self.t_phase = rospy.Time.now()
        self.done = False
        self.in_circle = False
        self.first_step = True     # zeroes the policy's prev_action on episode start
        self.push_done = False     # set by the action done_cb when a goal finishes
        self.preempt_sent = False  # debounce cancel_goal within one PUSH phase
        self.infer_pending = False
        self.settled_since = None  # start of the current run of settled samples
        self.steps = 0             # macro-steps dispatched this episode
        self.gate = None           # pending operator prompt, or None

        self.client = actionlib.SimpleActionClient("push", ExecutePushAction)
        rospy.loginfo("Waiting for ExecutePush action server...")
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            rospy.logwarn("ExecutePush server not up yet; will keep trying on first push.")
        self.infer = rospy.ServiceProxy("/npm/infer", InferPush)

        self.state_pub = rospy.Publisher("/npm/macro_state", MacroState, queue_size=1)
        self.marker_pub = rospy.Publisher("/npm/push_marker", Marker, queue_size=3)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate_hz), self._tick)
        rospy.loginfo("coordinator up (dwell=%.1fs settle=%.2f/%.2f hold=%.1fs "
                      "success_r=%.2f max_steps=%s)",
                      self.push_dwell, self.settle_vel, self.settle_ang,
                      self.settle_hold, self.success_radius,
                      self.max_steps or "inf")
        if self.gate_armed:
            rospy.loginfo("operator gates ON (settle=%s pre_push=%s): pushes wait "
                          "for ENTER in this terminal.", self.confirm_settle,
                          self.confirm_pre_push)

    def _dist(self):
        return self.est.dist_dir()[0]

    def _settled(self):
        # No fresh state means no settle: a stale estimate is frozen, and frozen
        # velocities would read as settled and start a push on dead data.
        vel = self.est.velocities()
        quiet = False
        if vel is not None:
            lin_vel, ang_vel = vel
            quiet = (float(np.linalg.norm(lin_vel)) < self.settle_vel and
                     float(np.linalg.norm(ang_vel)) < self.settle_ang)
        if not quiet:
            self.settled_since = None
            return False
        now = rospy.Time.now()
        if self.settled_since is None:
            self.settled_since = now
        return (now - self.settled_since).to_sec() >= self.settle_hold

    def _goto(self, phase):
        self.phase = phase
        self.t_phase = rospy.Time.now()
        # A settle run from the previous macro-step must not count toward this one.
        self.settled_since = None
        rospy.loginfo("phase -> %s (dist=%.3f)", NAMES[phase], self._dist())

    def _elapsed(self):
        return (rospy.Time.now() - self.t_phase).to_sec()

    def _request_push(self):
        # Called off the timer thread: a slow inference must not stall the state
        # machine, which still has to watch for success and time-outs.
        state = self.est.latest()
        if state is None:
            self.infer_pending = False
            return
        self.infer_pending = True
        threading.Thread(target=self._infer_worker, args=(state,),
                         daemon=True).start()

    def _infer_worker(self, state):
        req = InferPushRequest(state=state, reset=self.first_step)
        try:
            rospy.wait_for_service("/npm/infer", timeout=self.infer_timeout)
            resp = self.infer(req)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn_throttle(2.0, "coordinator: /npm/infer failed: %s", e)
            self.infer_pending = False
            return
        if not resp.success:
            rospy.logwarn("coordinator: policy refused: %s", resp.message)
            self.infer_pending = False
            return
        self.cmd = resp.command
        self.first_step = False
        self.infer_pending = False

    def _send_push(self, cmd):
        # Forward the policy's action as an ExecutePush goal. False = not sent.
        if cmd.loc_idx < 0:
            rospy.logerr("coordinator: policy returned loc_idx=%d, which is the "
                         "push_point sentinel, not an index. Not pushing.",
                         cmd.loc_idx)
            return False
        goal = ExecutePushGoal()
        goal.loc_idx = cmd.loc_idx
        goal.force_body = cmd.force_body     # object-frame locked force
        goal.duration = self.push_dwell
        goal.root_frame = ""                 # executor default (odom)
        self.push_done = False
        self.preempt_sent = False
        self.steps += 1
        self.client.send_goal(goal, done_cb=self._push_done_cb)
        return True

    def _push_done_cb(self, status, result):
        self.push_done = True
        name = _STATUS_NAMES.get(status, str(status))
        if result is None:
            rospy.logwarn("push done: status=%s with no result", name)
            return
        log = rospy.loginfo if status == GoalStatus.SUCCEEDED else rospy.logwarn
        # Never auto-retry a failed push (CLAUDE.md safety): the state machine
        # settles and re-infers on the state the failure actually left behind.
        log("push done: status=%s reason=%s contact=%s peak=%.1fN drift=%.3fm",
            name, result.end_reason, result.contact_made, result.peak_force,
            result.finger_travel)

    def _arm_gate(self, title, lines, allow_skip):
        keys = ["ENTER = continue"]
        if allow_skip:
            keys.append("s = skip, re-infer")
        keys.append("q = end episode")
        banner = ["", "=" * 68, " " + title]
        banner += ["   " + ln for ln in lines]
        banner += [" " + "   ".join(keys), "=" * 68]
        self.gate = _Gate("\n".join(banner), allow_skip)

    def _clear_gate(self):
        # The worker is a daemon blocked on input(); dropping the reference is
        # enough, its answer is simply never read.
        self.gate = None

    def _preview_pose(self, cmd):
        # World contact and force from the CURRENT object pose, not the pose
        # inference ran on: the object can be nudged while the prompt is up, and a
        # marker drawn at a stale pose would point the operator at the wrong place.
        # push_point/force_body are the policy's object-frame copies and stay in
        # this node; the goal still carries only loc_idx.
        state = self.est.latest()
        if state is None:
            return el.as_xyz(cmd.contact_point), el.as_xyz(cmd.push_force)
        q = state.pose.orientation
        t = np.array([state.pose.position.x, state.pose.position.y,
                      state.pose.position.z])
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return el.world_contact_from_object(R, t, el.as_xyz(cmd.push_point),
                                            el.as_xyz(cmd.force_body))

    def _arm_push_gate(self, cmd):
        contact, force = self._preview_pose(cmd)
        mag = float(np.linalg.norm(force))
        d = force / max(mag, 1e-9)
        self._publish_preview(contact, force, mag, cmd.loc_idx)
        rospy.loginfo("PROPOSED loc=%d |F|=%.1fN %s contact=(%.3f,%.3f,%.3f) "
                      "dir=(%.2f,%.2f,%.2f)", cmd.loc_idx, mag, self.world_frame,
                      contact[0], contact[1], contact[2], d[0], d[1], d[2])
        self._arm_gate(
            "APPROVE PUSH?    loc_idx = %d" % cmd.loc_idx,
            ["at %s (%.3f, %.3f, %.3f)"
             % (self.world_frame, contact[0], contact[1], contact[2]),
             "along (%.2f, %.2f, %.2f)   |F| = %.1f N"
             % (d[0], d[1], d[2], mag),
             "see the amber arrow in rviz"],
            allow_skip=True)

    def _settle_gate_or_leave(self, why):
        if not self.confirm_settle:
            self._leave_settle()
            return
        self._arm_gate("OBJECT SETTLED?  (%s)" % why,
                       ["dist to goal = %.3f m" % self._dist(),
                        "the next state sample is what the policy infers on"],
                       allow_skip=False)

    def _leave_settle(self):
        if self.in_circle:
            self._finish()
        else:
            self._goto(INFER)

    def _dispatch(self, cmd):
        self.cmd = None                    # one goal per macro-step
        if self._send_push(cmd):
            self._goto(PUSH)
        else:
            self._give_up()

    def _tick(self, _evt):
        dist = self._dist()
        if not self.done:
            self.in_circle = (not np.isnan(dist)) and dist < self.success_radius

        if self.done:
            pass
        elif self.phase == IDLE:
            if self.est.ready():
                self._goto(INFER)
        elif self.phase == INFER:
            # in_circle stays ahead of the gate: an object nudged into the goal
            # while the operator is deciding ends the episode, it is not pushed.
            if self.in_circle:
                self._finish()
            elif self.gate is not None:
                reply = self.gate.reply()
                if reply is None:
                    pass
                elif reply == "q":
                    self._clear_gate()
                    self._clear_preview()
                    self._give_up()
                elif reply == "s":
                    rospy.loginfo("push skipped by operator; re-inferring.")
                    self._clear_gate()
                    self._clear_preview()
                    self.cmd = None
                else:
                    self._clear_gate()
                    self._clear_preview()
                    self._dispatch(self.cmd)
            elif self.max_steps and self.steps >= self.max_steps:
                self._give_up()
            elif self.cmd is not None:
                if self.confirm_pre_push:
                    self._arm_push_gate(self.cmd)
                else:
                    self._dispatch(self.cmd)
            elif not self.infer_pending:
                self._request_push()
        elif self.phase == PUSH:
            # Preempt an in-flight push the moment the object reaches the goal circle.
            if self.in_circle and not self.preempt_sent:
                rospy.loginfo("Success mid-push - preempting goal.")
                self.client.cancel_goal()
                self.preempt_sent = True
            if self.push_done:
                self._goto(SETTLE)
        elif self.phase == SETTLE:
            if self.gate is not None:
                # _settled() and the timeout are frozen while the operator
                # decides, or the timeout would re-fire under its own prompt.
                reply = self.gate.reply()
                if reply is None:
                    pass
                elif reply == "q":
                    self._clear_gate()
                    self._give_up()
                else:
                    self._clear_gate()
                    self._leave_settle()
            elif self._settled():
                self._settle_gate_or_leave("velocity settled")
            elif self._elapsed() >= self.settle_timeout:
                self._settle_gate_or_leave("timeout %.1fs" % self.settle_timeout)

        self._publish_state(dist)

    def _give_up(self):
        rospy.logwarn("STOPPING after %d macro-steps without success (dist=%.3f)",
                      self.steps, self._dist())
        self._clear_gate()
        self._clear_preview()
        self.done = True
        self.phase = IDLE

    def _finish(self):
        self._clear_gate()
        self._clear_preview()
        if not self.done:
            rospy.loginfo("SUCCESS: object within %.2f m of goal (dist=%.3f)",
                          self.success_radius, self._dist())
        self.done = True
        self.in_circle = True
        self.phase = IDLE
        rospy.loginfo("episode ended after %d macro-steps", self.steps)

    def _publish_state(self, dist):
        m = MacroState()
        m.header.stamp = rospy.Time.now()
        m.phase = self.phase
        m.dist_to_goal = 0.0 if np.isnan(dist) else dist
        m.in_success_circle = self.in_circle
        self.state_pub.publish(m)

    def _publish_preview(self, contact, force, mag, loc_idx):
        # Amber in its own namespace: "proposed, not yet approved", so it reads as
        # distinct from the executor's green committed markers on the same topic.
        stamp = rospy.Time.now()
        scale = (self.arrow_len * min(mag, el.FORCE_NORM_CLIP) / el.FORCE_NORM_CLIP) \
            / max(mag, 1e-6)
        amber = ColorRGBA(1.0, 0.75, 0.1, 1.0)

        arrow = self._preview_marker(0, Marker.ARROW, stamp)
        arrow.points = [Point(*[float(v) for v in contact]),
                        Point(*[float(contact[i] + scale * force[i])
                                for i in range(3)])]
        arrow.scale.x, arrow.scale.y = 0.01, 0.02
        arrow.color = amber

        sphere = self._preview_marker(1, Marker.SPHERE, stamp)
        sphere.pose.position = Point(*[float(v) for v in contact])
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.04
        sphere.color = amber

        text = self._preview_marker(2, Marker.TEXT_VIEW_FACING, stamp)
        text.pose.position = Point(float(contact[0]), float(contact[1]),
                                   float(contact[2]) + 0.18)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.06
        text.color = amber
        text.text = "PROPOSED loc=%d  |F|=%.0fN" % (loc_idx, mag)

        for mk in (arrow, sphere, text):
            self.marker_pub.publish(mk)

    def _clear_preview(self):
        # A preview left standing would sit beside the executor's committed marker
        # and read as a second, competing push point.
        stamp = rospy.Time.now()
        for mid in (0, 1, 2):
            mk = self._preview_marker(mid, Marker.ARROW, stamp)
            mk.action = Marker.DELETE
            self.marker_pub.publish(mk)

    def _preview_marker(self, mid, mtype, stamp):
        mk = Marker()
        mk.header.frame_id = self.world_frame
        mk.header.stamp = stamp
        mk.ns = self.PREVIEW_NS
        mk.id = mid
        mk.type = mtype
        mk.action = Marker.ADD
        mk.lifetime = rospy.Duration(0.0)
        return mk


def main():
    rospy.init_node("coordinator_node")
    try:
        Coordinator()
    except _GateSetupError:
        return                      # already logged, shutdown already requested
    rospy.spin()


if __name__ == "__main__":
    main()

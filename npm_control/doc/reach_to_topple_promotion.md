# Promote reach assist -> topple mid-assist

Not implemented. This is the recipe.

## Problem

`_body_assist` picks `mode` once, from the caller, and never re-checks. An object that
starts tipping *after* the arm ran out of travel finishes the push in `reach` mode, which
honours the SMDP radius and the macro clock and so cuts the tip halfway.

Bag `spot_push_policy_3.bag`, `tilt_min_deg=10`: three reach assists during which the
object tipped to 7.9 / 22.3 / 25.8 deg with nothing watching. (The tilt-lag fix,
`~topple/tilt_root_frame`, already catches pushes 2 and 4 one tick earlier, in the push
loop. This promotion is the backstop for tips that begin after the handover.)

## Change

All in `scripts/executor_node.py`.

1. `_body_assist` signature: add `tilt_ref, tilt_root, tilt_gain`.
   Call site in `_push_loop` passes the three loop locals of the same names.

2. Top of `_body_assist`: `object_frame = self._object_frame(goal)`.

3. Replace the one-shot cap block with a helper, so it can be recomputed on promotion:

   ```python
   def travel_cap(mode, drift):
       if mode != "reach":
           return self.topple_max_body_travel
       return min(self.reach_max_body_travel,
                  max(0.0, self.max_finger_reach - drift) + 0.1)

   cap = travel_cap(mode, drift)
   ```

4. In the assist loop, after `drift` is updated and before `_publish_feedback`:

   ```python
   # Same gauge and same push-start reference the push loop used, so the tilt
   # trace is continuous across the handover.
   if tilt_ref is not None:
       R_tilt = self._tilt_rotation(object_frame, tilt_root)
       if R_tilt is not None:
           tilt_gain = el.tilt_since(tilt_ref, R_tilt)
   if mode == "reach" and tilt_gain > self.topple_tilt_min and latch.loaded:
       # The object started tipping after the mode was picked. Promote: a topple
       # gets the full travel budget and is exempt from the SMDP radius and the
       # macro clock, both of which would otherwise cut the tip halfway and leave
       # the object balanced on an edge.
       mode, end_reason = "topple", "topple_travel"
       cap = travel_cap(mode, drift)
       rospy.loginfo("ASSIST: reach -> topple, object tipped to %.1fdeg "
                     "(>%.1f) mid-assist; travel cap now %.2fm.",
                     np.rad2deg(tilt_gain), self.topple_tilt_min_deg, cap)
   ```

5. Add `tilt=` to the `assist=` debug line, so the bag stops going blind once an assist
   starts:

   ```python
   line = ("assist=%s tilt=%.2f/%.2fdeg travel=%.3f/%.3fm "
           "drift=%.3f/%.3fm F=%.1fN "
           "loaded=%d low=%.2f/%.2fs heading_err=%.0fdeg"
           % (mode, np.rad2deg(tilt_gain), self.topple_tilt_min_deg,
              travel, cap, drift, self.max_finger_reach, cur,
              latch.loaded, latch.low_for, self.contact_lost_grace,
              np.rad2deg(heading_err)))
   ```

6. Update the `_body_assist` docstring: delete the "mode is fixed for the lifetime of
   the assist" NOTE and this file's reference, say the promotion is one-way.

## Why nothing else moves

Every mode-dependent exit inside the loop already reads `mode` per tick, so promotion
switches them all with no further edits:

- `latch.lost` -> `"toppled"` instead of `"contact_lost"`
- `travel > cap` -> `"topple_travel"` instead of `"assist_travel"`
- the `if mode == "reach"` block (SMDP radius, macro deadline) stops running

`cap` only ever rises on promotion (`topple_max_body_travel` >= the reach cap), so travel
already accrued is safe to carry over.

## Test

- Bag replay: `spot_push_policy_3.bag`, feed the mocap object TF through `el.tilt_since`
  against the push-start reference, print the first assist tick clearing `tilt_min_deg`.
  Expect pushes 2/3/4 to promote, 0/1 never (true peaks 7.9 and 5.4 deg, under the gate).
- Bench `rosrun npm_control test_push_action_track.py`: hand-tip the object *after* the
  assist has started, look for `ASSIST: reach -> topple` and the raised cap.
- Regression: an untouched object must still end `assist_travel` / `reach` / `timeout`.

## Watch out

- Promotion must be one-way. `if mode == "reach"` gives that for free; do not add a
  demotion branch, or a tilt dropout would hand the macro clock back mid-tip.
- `tilt_ref` may be `None` (object TF never resolved at push start). Guard, as above.
- Right after the force -> position handover `latch.loaded` is briefly false by design
  (`contact_lost_grace`). Promotion just waits; do not relax the `latch.loaded` term to
  work around it.
- `end_reason` is reassigned at every real break, so setting it on promotion only covers
  the `rospy.is_shutdown()` exit. Keep it anyway.

# Recenter: base catch-up before the body-locked crawl

Implemented, `~topple/recenter` (default on, **topple mode only**).

## Problem

The arm-only push ends *at* the far-reach limit -- that is what `arm_out` /
`_tip_stalled` mean -- and the old `_body_assist` latched the strut pose right there.
An arm at full extension has no travel left to trade for wrist pitch, so it holds
position and loses orientation. The commanded pose is rooted in `flat_body`, which is
gravity-aligned, so a *tracked* latch would have held elevation flat; what is seen
instead is tracking error from an arm with nothing left to give.

Measured, bag `spot_push_policy_6.bag`, episode 1 (`PUSH` 7.0-9.1 s, `TOPPLE` 9.1-15.6
s), elevation of the tool x axis in `odom`:

| t (s) | phase | tool-x elev | hand z |
|---|---|---|---|
| 7.04 | push start | +20.0 deg (commanded force elev +20.3) | 1.078 |
| 9.11 | handover | +15.8 deg | 1.269 |
| 9.53 | topple +0.4 s | +20.0 deg | 1.277 |
| 12.80 | topple | +29.0 deg | 1.349 |
| 15.02 | topple end | +34.8 deg | 1.401 |

During the push the pitch tracks the tipping face *down* (`force_world = R_object @
force_body`, so the commanded task frame rotates with the object). Across the handover
it snaps back up ~4 deg -- the push runs `rx_axis = AXIS_MODE_FORCE`, so the measured
wrist roll that the old code latched was never a tracked setpoint -- and then climbs
19 deg over the assist while the hand rises 13 cm. All 13 pushes in that bag end in an
assist. Tip reach sat at ~1.00 m for the whole crawl.

## Change

`_body_assist` gains a first stage, `_recenter`:

1. Latch the **measured** tip position in `root` (odom). Command it all-position with
   `el.build_body_locked_arm_command(..., root_frame=root,
   remain_near_current_joints=False)` -- an odom-rooted target stays put while the base
   advances, so the arm folds back in. The flag is off because the arm has to re-solve
   continuously as it folds, which is what the flag damps.
2. Crawl at `recenter_body_vel` until the arm is back inside its envelope
   (`_recenter_gauge`, the mirror of `_reach_gauge`: reach <= `recenter_target_reach`).
   Reach shrinks ~1:1 with body travel along the push axis.
3. Orientation is **not** held. `_track_quat` aims the tool x axis at the live
   `force_world`, keeps the *measured* roll (swing-twist, so an all-position command
   cannot snap the compliant roll), bounds the elevation to the push-start value
   +- `recenter_pitch_band` and slew-limits it to `recenter_pitch_rate`.
4. On exit, re-latch `flat_body` tip pose, travel origin and orientation from the
   **post-recenter** configuration, and hand what recenter *commanded* -- not what the
   arm settled at -- to the crawl. With `recenter_track_orientation` the crawl keeps
   tracking too: position stays frozen (an object must never be able to drag the hand),
   orientation follows the face down as it tips.

Cost is zero in object terms: the tip is pinned in world, so the object is neither
pushed nor released while the base catches up.

## Exits

Aborts (return from the assist): `contact_lost`, `toppled`, `watchdog`, `preempted`.
Everything else -- `gauge`, `clearance`, `travel`, `floor`, `timeout` -- continues into
the crawl and warns, because a short recenter still leaves the arm better off than none.
`reach` mode does not recenter: it would spend body travel without advancing the push,
and a slide has no tipping face whose pitch has to be held.

## Clearance gate

`recenter_max_travel_cap` is a ceiling, not the operating limit. The live limit is
`el.body_clearance_to_cloud`: the object's collision cloud in `flat_body` against an
axis-aligned body box, the nearest blocking point's x minus the front face. Points
above `half_height + over_margin` are an overhang Spot can crawl *under* and block
nothing; likewise below the belly. Recomputed every tick, so a tipping object updates
it, and applied to the crawl too under `recenter_clearance_gate`.

The cloud is the `.obj` beside the `.npz` (`~npz_path` with the suffix swapped -- the
two files are one asset, so there is no separate parameter to point at the wrong mesh),
parsed by `el.load_mesh_cloud` into vertices plus seeded area-weighted surface samples.
Verbatim link-frame metres: `pts_norm*scale + centroid` undoes exactly the
normalisation the npz stores, checked against the npz bbox at load (`Paralelopiped`
agrees to 0.4 mm). `pcl_shrink` is not applied -- that margin keeps the pointer argmax
off physical edges, and a collision test wants the real surface.

Replayed on that bag with the default box, clearance falls 0.40 m -> 0.00 m across the
assist and crosses zero exactly as the crawl ends on its travel cap, with ~330 blocking
points throughout.

## Visualisation

`/npm/executor/body_box` (`CUBE` + clearance text, green/red/grey for
positive/blocked/no-data) and `/npm/executor/clearance_pts` (the blocking points),
both in `flat_body` at `~clearance/viz_rate`, and once per goal before anything moves
(including on the `dry_run` path). Displays are in both `npm_launch` rviz configs.

## Not done

- `reach` mode still latches at whatever extension the push ended in.
- The mesh is loaded once at startup, so a goal naming a different object than
  `~npz_path` gates on the wrong geometry. Same assumption as `link_pts`.
- Measured tip reach in the bag sits at ~1.00-1.06 m, above `el.shoulder_reach`'s
  0.9906 m geometric maximum, so either the ROS `hand` frame or `tool_tip_x` is a
  couple of cm off the model the gauge assumes. It does not change which side of
  `max_tip_reach` / `recenter_target_reach` the arm is on, but it should be pinned down
  before either threshold is tuned finely.

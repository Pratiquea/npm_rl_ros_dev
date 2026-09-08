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

1. Latch the **measured** tip position in `root` (odom), and hold *that world point*
   while the base advances, so the arm folds back in. It is commanded all-position in
   **`flat_body`**, re-derived from the fixed odom point through `flat_T_root` every
   tick: `el.build_body_locked_arm_command(..., remain_near_current_joints=False)`.
   The flag is off because the arm has to re-solve continuously as it folds, which is
   what the flag damps.

   The obvious form -- `root_frame=root`, one fixed odom target, no per-tick
   arithmetic -- is what this stage shipped with, and it does not work: an
   odom-rooted arm Cartesian command in the same `SynchronizedCommand` leaves the
   **base planted**, so nothing folds. See "The base never crawled" below.
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

## The base never crawled (2026-09-07)

Bag `spot_push_policy_10` push 3 ran the whole stage without moving: `travel=0.001 m`
over the full 3.0 s timeout at a commanded 0.10 m/s, `reach` 1.052 -> 1.052 m,
`reason=timeout`. Three independent measurements agree it did nothing at all:

| | body travel along body +x | feet | arm joints |
|---|---|---|---|
| `_10` recenter, push 3 | **+0.001 m** / 3.4 s | never leave the ground | < 3 deg |
| `_10` crawl, push 3 | **+0.549 m** / 5.8 s | trotting from +0.36 s | 3--8 deg |
| `_8` recenter, push 1 | **+0.001 m** / 2.5 s (lateral -0.062) | -- | -- |
| `_8` recenter, push 2 | **-0.001 m** / 3.1 s (lateral -0.193) | -- | -- |

Body pose from `vision -> body` in the bags, projected on the body +x axis the stage
commands. So this is not new in `_10`: **the recenter crawl has never worked.** Bag
`_8` only looked healthy because the logged `travel` is a magnitude and sideways drift
filled it, while the reach it exited on came from the *stale* hold pose dragging the
arm backwards -- the artifact [recontact_hold.md](recontact_hold.md) removed. Taking
the artifact away left the stage with nothing.

The only difference between the two stages' commands is the arm's root frame: the
crawl roots in `flat_body` and walks, the recenter rooted in `odom` and did not. Both
carry the same kind of SE2 velocity sub-command. The fix keeps the physics and changes
the frame that carries it, per step 1 above. The exact SDK mechanism is **not
confirmed** -- what is measured is that the flat_body-rooted form walks and the
odom-rooted form does not.

If it turns out to be the `SynchronizedCommand` packaging rather than the whole-body
controller, the fallback is the pattern ALIGN already uses: an arm-only command plus a
*separate* mobility-only command at 2--3 Hz, which `el.build_se2_base_command` documents
as coexisting sub-commands. The change above corrects either condition, so try it first.

**Bench check, no object:** arm to full extension, run recenter alone for 3.0 s,
measure body movement along body +x. Correct is 0.25--0.30 m. The old code gives 0.001 m.

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

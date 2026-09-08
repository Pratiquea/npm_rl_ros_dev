# Base aim: pointing the body at the object's yaw pivot

## The problem

The body assist crawls the base forward with the arm locked as a rigid strut.
The legs' force therefore acts at the contact point, along the **body +x axis**.
Its moment about the object's vertical axis (the world-z line through the object,
not the object's own z) is

```
M_z = ((p_contact - p_pivot) x F)_z
```

so the object yaws unless that line of action passes through the pivot.

Observed on the robot, with the contact point relative to the object's vertical
axis:

| contact | object during the assist | result |
|---|---|---|
| left of the axis | yaws counterclockwise | contact slides off, chassis can reach the object first |
| through the axis | no yaw, pure topple/slide | works |
| right of the axis | yaws clockwise | same failure, mirrored |

The middle row is the one that worked, and it is the one where `push_dir` already
points at the pivot. `el.base_align_pose` used to face the base straight down
`push_dir`, so on the other two rows the crawl kept re-applying the same moment
that spun the object in the first place. Orientation tracking (see
[recenter_stage.md](recenter_stage.md)) cannot fix it: it steers the tool, and the
problem is where the **body** points.

## The change

Two stages, both driven by the same pivot.

### 1. ALIGN aims at the pivot (`~align/aim_mode: pivot`)

`el.support_pivot_xy` takes the object's collision cloud in a gravity-aligned
frame and returns the centroid of the points within `~align/support_eps` of the
lowest one - the support polygon. That single rule covers both cases the assist
sees:

- flat object: the whole footprint, which is where friction resists yaw;
- tipping object: only the leading edge, because everything else has left the
  floor, and that edge IS the topple pivot.

The link origin is not a substitute. `Paralelopiped`'s sits 0.2 m off its own
footprint centre.

`el.aim_dir_to_pivot` then returns the unit heading from contact to pivot, and
`_walk_base_align` walks the base to `base_standoff + align_standoff` behind the
contact **on that line**, so the contact still sits on the body x axis.

Guards, each falling back to the old push-axis heading and saying why:

| case | condition | log |
|---|---|---|
| no cloud / no object TF | `pivot is None` | `no pivot` |
| contact over the pivot | lever < `~align/min_lever` | `contact over pivot` |
| contact on the far face | `dot(aim, push_dir) <= 0` | `pivot behind contact` |
| aim too far off the push | `abs(ang) > ~align/aim_cone` | clamped to the cone, warns |
| near-vertical push | `abs(push_dir_xy) ~ 0` | pre-walk skipped, as before |

The cone matters because the arm's standoff pose is still built along `push_dir`:
at the 35 deg default it sits `align_standoff * sin(35) = 86 mm` off the body x
axis. For `Paralelopiped` a contact 0.10 m off centre needs 11.4 deg and one at
0.30 m needs 31 deg, so the cone binds only on extreme contacts.

### 2. Both assist stages steer back onto the live aim (`~topple/aim_track`)

ALIGN aims before the push, and the push then yaws the object for its whole
duration, so that heading is stale by the handover. `_aim_heading_err` recomputes
the aim every tick from the **live tip** - where the strut actually hands the
force over - and `_aim_yaw_rate` turns the error into a base yaw rate: P gain,
rate limit, deadband (the pivot comes from a tracked pose and a sampled cloud;
without the deadband a loaded base hunts on noise).

The two stages differ in what base yaw costs them:

- **recenter**: the tip is pinned in `odom`, so base yaw cannot move it. No
  budget; the stage's own timeout bounds the total.
- **crawl**: the tip is latched in `flat_body`. A yawing base would sweep a tip
  0.9 m ahead sideways across the pushed face - a sliding contact, not a push.
  `el.counter_yaw_tip` rotates the commanded offset back by the yaw accrued since
  the latch, so the tip keeps its **world bearing** from the body origin and only
  the body's *translation* drives it into the object. Verified in the mock
  harness: commanded tip bearing in `flat_body` tracks `-base_yaw` exactly, radius
  drift 2e-13 mm.

  The counter-rotation stops the contact sliding but the tip still ends up
  further off the body x axis with every radian, spending arm envelope the crawl
  is short of already. Hence `~topple/aim_yaw_max` (20 deg default), a cumulative
  cap that zeroes the rate in the direction that would spend more. In the harness,
  a 24.8 deg error closes to 4.9 deg under a 20 deg cap and stalls at 19.2 deg
  under a 3 deg cap - the cap doing exactly what it says.

## Observability

- `/npm/executor/aim` (`Marker`, ns `npm_aim`): magenta sphere at the pivot, cyan
  line contact->pivot (the aim), orange line for the reference it is compared
  against - the push axis at ALIGN, the body x axis during the assist. The number
  that matters is the angle between the last two, and no log line shows an angle.
  Published once per goal before anything moves, `dry_run` included, then
  throttled to `~clearance/viz_rate` through both assist stages.
- `assist_debug` gains `aim_err=`, `yaw=<accrued>/<cap>` and `v_rot=` on the crawl
  line, `aim_err=` and `v_rot=` on the recenter line. `aim_err=n/a` means there was
  nothing to aim at and no yaw was commanded.
- The one-shot handover warning now measures against the aim, not the push axis,
  and says which of the two it used.
- `/npm/executor/body_box` and `/npm/executor/clearance_pts` now use one rviz
  namespace each (`npm_body_box`, `npm_clearance_pts`). They were both
  `npm_clearance`, which gave the two displays the same checkbox label.

## Not done

- **Predicted-swing standoff.** Rotating the cloud about the pivot by +-psi at
  ALIGN and pushing the standoff back until the worst case still clears would
  catch a corner swinging into the body before the live gate does. Deferred.
- **Yaw lead.** Biasing the ALIGN heading by the yaw the push itself will add,
  i.e. running the ALIGN rule on the *predicted* pose at handover. The gain is not
  derivable a priori - it needs a fit of measured `Delta yaw` (push start ->
  assist start) against `M_z` across bags - and a wrong sign doubles the error
  instead of removing it. Deferred pending that data.
- **Reach mode.** Both stages run in topple and reach alike here, unlike the
  recenter, because a sliding object yaws too. Untested against a reach-mode bag.

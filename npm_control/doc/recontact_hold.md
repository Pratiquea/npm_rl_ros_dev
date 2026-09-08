# Recontact hold: latch the assist pose after the arm stops, not when the push ends

## The problem

The arm-only push runs in **force mode**. When the reach gauge trips, the tip is
still travelling — 0.5 to 1.7 m/s measured across `spot_push_policy_8` and `_9`.

Everything downstream then latched its hold pose out of the `snap` that the push
loop took at the **top** of that tick, which by the time the hold command went out
was 40–160 ms old. So the arm coasted past the pose it was about to be told to
hold, and the position hold pulled it **back**.

The retract, from the mocap tip projected on its own travel direction at the
handover:

| bag | aim tracking | retract per push |
|---|---|---|
| `spot_push_policy_6` (pre-recenter) | n/a | 0.000, 0.000, 0.001, 0.002, 0.005, 0.006, 0.006, 0.006, 0.008, 0.028 m |
| `spot_push_policy_8` | on | 0.023, 0.069, 0.086, 0.087, 0.136, 0.207 m |
| `spot_push_policy_9` | off | 0.165, 0.181, 0.192, 0.233, 0.262, 0.280, 0.328, 0.573 m |

Aim tracking is not the cause — it is on in one bag and off in the other. Bag 6
predates the recenter stage and does not have it, so the regression arrived with
RECENTER (wiki log, 2026-09-05): before it, the same stale snapshot fed a
`flat_body` latch and the crawl immediately walked the error off; after it, the
arm holds a **world-fixed** point for up to `recenter_timeout` and the step is
fully expressed and then held.

The executor's own logs measure the staleness directly — latched reach against the
reach read on the first recenter tick, 0.04–0.16 s later:

```
bag 8:  +0.079  +0.055  +0.080  +0.025  +0.057  +0.072 m
bag 9:  +0.088  +0.048  +0.078  +0.051 m
```

## Why it matters

The retract lands inside the window where the object has tipped away and is
touching nothing. Sequence, `spot_push_policy_8` episode 1:

```
14.40  arm at 1.26 m/s, F=75N              <- the snapshot that got latched
14.46  PHASE -> TOPPLE, recenter latches it
14.48  tilt 25deg, tip at its forward extreme
14.57  F 50 -> 7.8N   contact lost, tilt 30deg      tip starts moving back
15.07  F 0.7N         tilt 35deg (peak)             tip -0.14 m, still moving
15.57  tilt 35 -> 16.7deg: the object is falling back
15.73  F 24N          contact regained ON THE RETRACTED TIP
15.9-17.3  tip static within 1 mm while the body crawls
```

Tip offset at the moment contact came back, relative to the handover: **-0.118,
-0.115, -0.130, -0.239 m** across bags 8/9. On bag 6 it is zero or positive.

## The stage

`_await_recontact` runs first in a topple assist, ahead of RECENTER. It stands
still — base planted, tip pinned in `root` — so it costs the assist nothing in
object terms: a tip that does not move neither pushes the object nor releases it.
Only the orientation moves, tracking the tipping face through `_track_quat`
exactly as the recenter does.

Two fixes, both here:

1. The pose comes from a state read taken **now**, after the arm has stopped, so
   commanding it is a no-op instead of a step backwards.
2. The pose handed on to RECENTER and the crawl is **re-read at the instant
   contact returns**, not at the instant the push ended.

Two ways in, decided by the force at entry:

| at the handover | starts in | then |
|---|---|---|
| loaded (usual — the bags show 15–75 N still on the tool) | `wait_break` | on `latch.lost`, switch to `wait_return` |
| already unloaded | `wait_return` | — |

Exits:

| exit | condition | result |
|---|---|---|
| `recontact` | force crossed `contact_made_n` after the break | **re-latch** the measured tip and the tracked orientation; this is the assist's hold pose |
| `held` | loaded continuously for `recontact_settle` with no break | keep the entry pose — the object never left, so there is nothing to wait for |
| `toppled` | `recontact_wait` expired while still waiting | ABORT. The object went over; crawling now would drive the body at empty air |
| `watchdog` / `preempted` / `error` | as everywhere else | ABORT |

### Two contact latches, and a third

`ContactLatch.made` and `.lost` are both sticky, so one latch cannot report a loss
and then a fresh make. `latch` watches the break-away, `back` (created at the
break) watches the return.

The assist's *shared* latch is **replaced** on the way out, and `entered` with it.
The break-away this stage waits through is expected, not a collapse: letting it
reach the latch the crawl exits on would abort the assist at the first recenter
tick, and `topple_load_grace` should run from the re-contact, not from the
handover.

The replacement starts **armed** (`latch.force_made`). This was missed at first, and
it killed pushes 1 and 2 of bag `spot_push_policy_10` inside recenter, as
`end_reason="contact_lost"` on an object that had never left the tool:

| | threshold | value |
|---|---|---|
| RECONTACT counts load at | `contact_eps` | 3.0 N |
| a *fresh* `ContactLatch` arms at | `contact_made_n` | 5.0 N |
| an object settled on a stationary tool sits at | -- | 3--5 N |

The band between the two is hysteresis, and hysteresis only reads as contact to a
latch that is already made. Re-arming from zero demands a re-load that a settled
object never delivers, so the stage quits on its own handover after
`topple_load_grace`. Push 1 held 4.0--4.3 N for 0.6 s and died; push 2 held
1.5--2.5 N and died; push 3 survived only because its load happened to drift over
5 N. Raising `topple_load_grace` does **not** help -- the load is flat, not rising.

Arming by fiat is a statement of measured fact, not an assumption: this stage
returns without an abort only when contact is confirmed present (`held` is
`recontact_settle` of continuous load, `recontact` is `back.made`). A real collapse
still ends the assist -- below `contact_eps` for `contact_lost_grace` sets
`latch.lost`, which reads as `toppled`.

Note the asymmetry this stage introduced. Before it, the assist began while the push
force was still decaying and the tip carried 11--42 N, so a fresh latch armed on its
first tick. Standing still for 0.6--1.3 s is exactly what lets the load fall into the
hysteresis band. The stage manufactured the condition its own latch reset could not
survive.

### Params

| param | default | note |
|---|---|---|
| `~topple/recontact` | `true` | `false` restores the old behaviour exactly — the rollback switch |
| `~topple/recontact_wait` | 2.5 s | break-away **plus** fall-back. The bags lose contact 0.04–0.99 s after the handover and get it back 0.5–2.1 s after it, so 1.0 s would cut off the slowest returns |
| `~topple/recontact_settle` | 0.5 s | continuous load that ends the stage early when there was no break-away. Without it every push pays the whole budget standing still |

`_recenter` now takes `tip_root`/`q_cmd` as arguments. Deriving them internally is
the `recontact:false` fallback, and it is exactly what produced the backward step.

### Observability

`stage=recontact` lines on `/npm/executor/assist_debug`, with `state=wait_break` /
`state=wait_return`, the held point, live reach, clearance, force, the
continuously-loaded timer and the budget. One summary line per stage naming the
reason, the break time and the reach either side.

## Verification

Offline only. **Nothing has run on the robot.**

Mock harness (`mock_recenter_test.py`), with a scripted force profile and a
snapshot named `"stale"` that reports the arm 0.15 m short of where it is:

- breaks away at 0.55 s and returns at 0.75 s -> `recontact`
- breaks away and never returns -> abort `toppled` at the 2.5 s budget
- never breaks away -> `held` at `recontact_settle`
- already unloaded at the handover -> `recontact`
- **the regression itself**: first odom-rooted commanded tip x, arm really at
  1.253 m, stale snapshot claiming 1.109 m —

  ```
  recontact ON : 1.253 m   (step   +0 mm)
  recontact OFF: 1.109 m   (step -144 mm)
  ```

All the pre-existing recenter, crawl and aim scenarios still pass unchanged.

## Not done

- **The push is still cut short of the real arm limit.** All 14 handovers in bags
  8 and 9 fire on `arm_out=1` — never the stall gauge, never drift — at
  `shoulder_radius` 0.941–1.030 against a 0.930 limit, while the reach measured
  one tick later is 1.030–1.066. The gauge trips roughly 0.1 m early. That is the
  `el.shoulder_reach` 0.9906 m geometric max vs measured tip reach discrepancy
  already flagged in the 2026-09-05 log entry; fix it by measuring the `hand`
  frame / `tool_tip_x` offset, not by moving the threshold. A COAST sub-stage that
  keeps the force command running until the tip genuinely stalls was designed and
  **deliberately not implemented** — it puts the arm in force mode against free
  air, which can drive it into a joint stop.
- The re-latch takes the **measured** tip at re-contact, so a hard landing would
  bake its deflection into the hold. At `contact_made_n` = 5 N the deflection is
  small; worth watching on the first bench run.
- After the hold the arm is still near full extension when the object lands on it.
  The old accidental retract happened to give the joints some room. If the impact
  reads high on the wrench estimate, add a small deliberate standoff at the latch.

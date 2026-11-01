"""What a session's motion data says about where the brush was.

The handle never says where in the mouth it is. It only says how it is held,
through the accelerometer and gyroscope frames on the SensorData stream, so a
zone can only be recognised if each one is brushed with the handle held
differently enough from the others. Whether that holds for a given person is
what this module measures - it does not place the brush yet.

A guided routine makes the measurement possible without anyone labelling
anything: the handle buzzes to move the user on, so as long as the user
follows it, the pacing step says which zone is being brushed. Each step is
reduced to the direction gravity pulls in from the handle's point of view,
and the steps are then compared.

The input is a recording: a list of events in the JSONL shape written by
``scripts/sonicare_session_record.py`` and by the integration's own motion
recording, one characteristic value per event. Both sources share this code,
so a recording made either way is read the same way.

Plain Python only - no Home Assistant import - so the command-line script in
``scripts/`` can load it without a Home Assistant install.

The raw values have no known unit. Nothing here depends on one: an attitude
is a direction, and the length of the vector is divided out.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .const import (
    BRUSHING_MODES,
    SENSOR_FRAME_GYROSCOPE,
    brushing_mode_for_model,
    sector_sequence,
    sector_step_seconds,
    uses_routine_id_mode,
)

HANDLE_STATE_RUN = 2

# The interpolated session clock is only trusted this far past the last
# brushing-time value. The handle counts whole seconds, so a little over one
# is the most that can pass between two of them while the motor runs.
CLOCK_SLACK_S = 1.5

# Which jaw each zone of the six-zone layout lies in: the handle paces
# upper right, upper front, upper left, then lower left, lower front, lower
# right. The four-zone Kids layout is left out - its zones have not been
# checked against a recording.
ZONE_JAW = {1: "upper", 2: "upper", 3: "upper",
            4: "lower", 5: "lower", 6: "lower"}

# The jaw is judged per second of brushing, not per frame. A single frame
# can catch the handle mid-stroke; a second of them averages that out.
JAW_WINDOW_S = 1.0


@dataclass
class Frame:
    t: float
    counter: int
    acc: tuple[int, int, int]
    gyro: tuple[int, int, int]
    elapsed: float | None = None
    step: int | None = None
    zone: int | None = None


@dataclass
class Session:
    model: str | None
    mode: str | None
    routine_length: int | None
    frames: list[Frame] = field(default_factory=list)
    pressure_frames: int = 0
    # Raw quadpacer readings with their timestamps. What this characteristic
    # reports has never been confirmed, so it is shown next to the derived
    # step rather than used.
    quadpacer: list[tuple[float, int]] = field(default_factory=list)


def _u(hex_str: str) -> int:
    """A little-endian unsigned value of whatever length the handle sent."""
    return int.from_bytes(bytes.fromhex(hex_str), "little")


def parse_gyro_frame(hex_str: str) -> tuple[int, tuple, tuple] | None:
    """Return (counter, acc, gyro) for a motion frame, None for anything else.

    The frame carries the gyroscope first and the accelerometer second, as
    ``scripts/sonicare_scan.py`` decodes it. The accelerometer is the triplet whose magnitude stays near one g
    (about 4096) the whole session; the gyroscope's sits near zero and spikes
    when the handle is turned.
    """
    raw = bytes.fromhex(hex_str)
    if len(raw) < 16:
        return None
    frame_type, counter = struct.unpack("<HH", raw[:4])
    if frame_type != SENSOR_FRAME_GYROSCOPE:
        return None
    values = struct.unpack("<6h", raw[4:16])
    return counter, values[3:], values[:3]


def read_recording(path: Path) -> tuple[dict, list[dict]]:
    """Return the meta line and the value events of a JSONL recording."""
    meta: dict = {}
    events: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("kind") == "meta":
                meta = event
            elif "hex" in event and event.get("char"):
                events.append(event)
    return meta, events


def load_session(path: Path, mode_override: str | None = None) -> Session:
    """Read a recording file and put every motion frame on the session clock."""
    meta, events = read_recording(path)
    return session_from_events(meta, events, mode_override)


def session_from_events(
    meta: dict, events: Iterable[dict], mode_override: str | None = None
) -> Session:
    """Put every motion frame of a recording on the session clock."""
    model = meta.get("model")
    session = Session(
        model=model,
        mode=mode_override or meta.get("mode"),
        routine_length=meta.get("routine_length"),
    )
    # The mode a recording names in its meta is the one the integration had
    # decoded. A raw value in the events is decoded here only when the meta
    # has none - the script's recordings, which carry the raw value alone.
    decode_mode = session.mode is None
    routine_id_mode = uses_routine_id_mode(model or "")

    handle_state: int | None = None
    clock: tuple[float, int] | None = None  # (t, brushing_time) of last value

    for event in sorted(events, key=lambda e: e["t"]):
        name, hex_str, t = event["char"], event["hex"], event["t"]
        if not hex_str:
            continue
        if name == "handle_state":
            handle_state = _u(hex_str[:2])
        elif name == "brushing_time":
            clock = (t, _u(hex_str))
        elif name == "routine_length":
            session.routine_length = _u(hex_str) or session.routine_length
        elif name == "quadpacer_segment":
            session.quadpacer.append((t, _u(hex_str)))
        elif name == "brushing_mode" and decode_mode and not routine_id_mode:
            session.mode = brushing_mode_for_model(model or "", _u(hex_str[:2]))
        elif name == "available_routine_ids" and decode_mode and routine_id_mode:
            session.mode = BRUSHING_MODES.get(_u(hex_str[:2]))
        elif name == "sensor_data":
            parsed = parse_gyro_frame(hex_str)
            if parsed is None:
                session.pressure_frames += 1
                continue
            counter, acc, gyro = parsed
            frame = Frame(t=t, counter=counter, acc=acc, gyro=gyro)
            if handle_state == HANDLE_STATE_RUN and clock is not None:
                since = t - clock[0]
                if since <= CLOCK_SLACK_S:
                    frame.elapsed = clock[1] + since
            session.frames.append(frame)

    _assign_steps(session)
    return session


def _assign_steps(session: Session) -> None:
    """Label each frame with the pacing step - and so the zone - it fell in."""
    steps = sector_step_seconds(
        session.model or "", session.mode, session.routine_length)
    if not steps:
        return
    sequence = sector_sequence(session.model or "", session.mode) or []
    step_len = steps[0]
    for frame in session.frames:
        if frame.elapsed is None:
            continue
        idx = int(frame.elapsed // step_len)
        if idx >= len(steps):
            continue
        frame.step = idx
        frame.zone = sequence[idx] if idx < len(sequence) else None


# ── Geometry ────────────────────────────────────────────────────────────────

def _unit(v) -> tuple[float, float, float] | None:
    n = math.sqrt(sum(c * c for c in v))
    if n == 0:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def mean_direction(vectors) -> tuple[float, float, float] | None:
    """The average direction of a set of vectors, each counted once.

    Normalising first keeps a hard tap from outweighing a second of steady
    brushing. What the motor adds averages out: it shakes the handle back and
    forth around the attitude it is held in.
    """
    units = [u for u in (_unit(v) for v in vectors) if u]
    if not units:
        return None
    return _unit(tuple(sum(u[i] for u in units) for i in range(3)))


def angle_deg(a, b) -> float:
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def spread_deg(vectors, centre) -> float:
    """Root-mean-square angle of a set of vectors around their mean."""
    angles = [angle_deg(u, centre) for u in (_unit(v) for v in vectors) if u]
    if not angles:
        return 0.0
    return math.sqrt(sum(a * a for a in angles) / len(angles))


def pitch_roll(g) -> tuple[float, float]:
    """Gravity as pitch and roll, the way an attitude is usually written."""
    x, y, z = g
    pitch = math.degrees(math.atan2(-x, math.sqrt(y * y + z * z)))
    roll = math.degrees(math.atan2(y, z))
    return pitch, roll


# ── Measures ────────────────────────────────────────────────────────────────

def stream_stats(session: Session) -> dict:
    frames = session.frames
    if len(frames) < 2:
        return {"frames": len(frames)}
    span = frames[-1].t - frames[0].t
    # The counter is 16 bits and wraps; anything it skipped was sent and lost.
    lost = 0
    for prev, cur in zip(frames, frames[1:]):
        gap = (cur.counter - prev.counter) % 0x10000
        if gap > 1:
            lost += gap - 1
    running = [f for f in frames if f.elapsed is not None]
    run_span = (running[-1].t - running[0].t) if len(running) > 1 else 0
    return {
        "frames": len(frames),
        "span_s": span,
        "rate_hz": (len(frames) - 1) / span if span else None,
        "running_frames": len(running),
        "running_rate_hz": (len(running) - 1) / run_span if run_span else None,
        "lost": lost,
        "lost_pct": 100 * lost / (len(frames) + lost),
    }


def step_table(session: Session) -> list[dict]:
    rows = []
    by_step: dict[int, list[Frame]] = {}
    for frame in session.frames:
        if frame.step is not None:
            by_step.setdefault(frame.step, []).append(frame)
    for step in sorted(by_step):
        frames = by_step[step]
        g = mean_direction(f.acc for f in frames)
        if g is None:
            continue
        pitch, roll = pitch_roll(g)
        t0, t1 = frames[0].t, frames[-1].t
        quad = sorted({v for t, v in session.quadpacer if t0 <= t <= t1})
        rows.append({
            "step": step + 1,
            "zone": frames[0].zone,
            "frames": len(frames),
            "gravity": g,
            "pitch": pitch,
            "roll": roll,
            "spread": spread_deg((f.acc for f in frames), g),
            "motion": sum(math.sqrt(sum(c * c for c in f.gyro))
                          for f in frames) / len(frames),
            "quadpacer": quad,
            "_frames": frames,
        })
    return rows


def jaw_of(acc_sum) -> str:
    """Upper or lower jaw, from the accelerometer's z axis.

    The z axis is the one the bristles face along. On the upper teeth they
    point up, so gravity reads positive there; on the lower teeth they point
    down. That follows from how a brush has to be held to reach a jaw at all,
    not from how one person holds it, so the rule is fixed: nothing is learned
    from earlier sessions and nothing is adjusted while brushing.
    """
    return "upper" if acc_sum[2] > 0 else "lower"


def jaw_coverage(session: Session) -> dict[str, Any] | None:
    """Seconds spent on each jaw, measured, next to what the routine asked.

    Judged after the session, one second of brushing at a time. Each pacing
    step also gets the share of its seconds spent on the jaw its zone lies
    in, which is where a step that was cut short or brushed in the wrong
    place shows up.
    """
    windows: dict[int, list[Frame]] = {}
    for frame in session.frames:
        if frame.elapsed is not None:
            windows.setdefault(int(frame.elapsed // JAW_WINDOW_S), []).append(frame)
    if not windows:
        return None
    measured = {"upper": 0.0, "lower": 0.0}
    expected = {"upper": 0.0, "lower": 0.0}
    per_step: dict[int, list[int]] = {}  # step -> [on its jaw, total]
    zone_of_step: dict[int, int | None] = {}
    for frames in windows.values():
        acc_sum = [sum(f.acc[i] for f in frames) for i in range(3)]
        jaw = jaw_of(acc_sum)
        measured[jaw] += JAW_WINDOW_S
        step, zone = frames[0].step, frames[0].zone
        target = ZONE_JAW.get(zone)
        if step is None or target is None:
            continue
        expected[target] += JAW_WINDOW_S
        zone_of_step[step] = zone
        counts = per_step.setdefault(step, [0, 0])
        counts[0] += jaw == target
        counts[1] += 1
    hits = sum(c[0] for c in per_step.values())
    total = sum(c[1] for c in per_step.values())
    return {
        "upper_s": measured["upper"],
        "lower_s": measured["lower"],
        "upper_expected_s": expected["upper"],
        "lower_expected_s": expected["lower"],
        "match_pct": 100 * hits / total if total else None,
        "steps": [
            {"step": s + 1, "zone": zone_of_step[s], "jaw": ZONE_JAW[zone_of_step[s]],
             "on_jaw_pct": 100 * c[0] / c[1]}
            for s, c in sorted(per_step.items())
        ],
    }


def zone_centres(rows: list[dict], part: str = "all") -> dict[int, tuple]:
    """Average attitude per zone, from all of each step or from one half."""
    vectors: dict[int, list] = {}
    for row in rows:
        if row["zone"] is None:
            continue
        frames = row["_frames"]
        half = len(frames) // 2
        chosen = {"all": frames, "first": frames[:half],
                  "second": frames[half:]}[part]
        vectors.setdefault(row["zone"], []).extend(f.acc for f in chosen)
    centres = {}
    for zone, vecs in vectors.items():
        c = mean_direction(vecs)
        if c is not None:
            centres[zone] = c
    return centres


def _classify(rows: list[dict], centres: dict[int, tuple], part: str):
    """Nearest-attitude classification of every frame in the given part."""
    hits = total = 0
    per_zone: dict[int, list[int]] = {}
    for row in rows:
        zone = row["zone"]
        if zone not in centres:
            continue
        frames = row["_frames"]
        chosen = frames[len(frames) // 2:] if part == "second" else frames
        for frame in chosen:
            u = _unit(frame.acc)
            if not u:
                continue
            guess = min(centres, key=lambda z: angle_deg(u, centres[z]))
            ok = guess == zone
            hits += ok
            total += 1
            per_zone.setdefault(zone, [0, 0])
            per_zone[zone][0] += ok
            per_zone[zone][1] += 1
    return hits, total, per_zone


def recognition(rows: list[dict]) -> tuple[float, dict[int, float]] | None:
    """Train on each step's first half, classify the second half's frames.

    Optimistic, because both halves come from the same session and the same
    hand - a random split of one recording is known to overstate what a
    classifier does on the next one. A zone that fails even this will not be
    recognised in a session it was not learned from; for the real figure see
    :func:`cross_session_recognition`.
    """
    centres = zone_centres(rows, "first")
    if len(centres) < 2:
        return None
    hits, total, per_zone = _classify(rows, centres, "second")
    if not total:
        return None
    return 100 * hits / total, {z: 100 * h / n for z, (h, n) in sorted(per_zone.items())}


def cross_session_recognition(
    sessions: list[Session],
) -> tuple[float, list[float]] | None:
    """Learn the zones from all other sessions, classify each one in turn.

    This is the figure that matters: whether yesterday's brushing says where
    the brush is today. Needs at least two sessions.
    """
    tables = [step_table(s) for s in sessions]
    tables = [t for t in tables if t]
    if len(tables) < 2:
        return None
    per_session: list[float] = []
    hits_all = total_all = 0
    for i, rows in enumerate(tables):
        training = [r for j, other in enumerate(tables) if j != i for r in other]
        centres = zone_centres(training)
        if len(centres) < 2:
            continue
        hits, total, _ = _classify(rows, centres, "all")
        if total:
            per_session.append(100 * hits / total)
            hits_all += hits
            total_all += total
    if not total_all:
        return None
    return 100 * hits_all / total_all, per_session


def closest_zones(centres: dict[int, tuple]) -> tuple[int, int, float] | None:
    zones = sorted(centres)
    pairs = [(a, b) for a in zones for b in zones if a < b]
    if not pairs:
        return None
    a, b = min(pairs, key=lambda p: angle_deg(centres[p[0]], centres[p[1]]))
    return a, b, angle_deg(centres[a], centres[b])


def summary(session: Session) -> dict[str, Any]:
    """The measures of one session, small and JSON-safe.

    Meant for an entity's attributes: everything needed to judge a session
    without its frames, including the zone attitudes, so that sessions can be
    compared later from the history of those attributes alone.
    """
    stats = stream_stats(session)
    out: dict[str, Any] = {
        "model": session.model,
        "mode": session.mode,
        "routine_length": session.routine_length,
        "frames": stats["frames"],
        "pressure_frames": session.pressure_frames,
    }
    if stats["frames"] < 2:
        return out
    out.update({
        "duration_s": round(stats["span_s"], 1),
        "rate_hz": round(stats["rate_hz"], 1) if stats["rate_hz"] else None,
        "running_frames": stats["running_frames"],
        "lost": stats["lost"],
        "lost_pct": round(stats["lost_pct"], 1),
    })
    rows = step_table(session)
    out["steps"] = [
        {
            "step": r["step"],
            "zone": r["zone"],
            "frames": r["frames"],
            "pitch": round(r["pitch"]),
            "roll": round(r["roll"]),
            "spread": round(r["spread"]),
            "motion": round(r["motion"]),
            "quadpacer": r["quadpacer"],
        }
        for r in rows
    ]
    centres = zone_centres(rows)
    out["zone_attitudes"] = {
        str(z): [round(c, 3) for c in g] for z, g in sorted(centres.items())
    }
    closest = closest_zones(centres)
    if closest:
        out["closest_zones"] = [closest[0], closest[1]]
        out["closest_angle"] = round(closest[2])
    rec = recognition(rows)
    if rec:
        out["recognition_pct"] = round(rec[0])
        out["recognition_by_zone"] = {str(z): round(p) for z, p in rec[1].items()}
    jaws = jaw_coverage(session)
    if jaws:
        out["jaws"] = {
            "upper_s": round(jaws["upper_s"]),
            "lower_s": round(jaws["lower_s"]),
            "upper_expected_s": round(jaws["upper_expected_s"]),
            "lower_expected_s": round(jaws["lower_expected_s"]),
            "match_pct": None if jaws["match_pct"] is None else round(jaws["match_pct"]),
            "steps": [{**s, "on_jaw_pct": round(s["on_jaw_pct"])} for s in jaws["steps"]],
        }
    return out


def format_report(session: Session) -> str:
    """The measures of one session as text, for a terminal."""
    lines: list[str] = []
    w = lines.append
    w(f"Model {session.model or '?'}  mode {session.mode or '?'}  "
      f"routine {session.routine_length or '?'} s")
    w("")

    stats = stream_stats(session)
    w("Stream")
    if stats["frames"] < 2:
        w(f"  {stats['frames']} motion frame(s) - nothing to analyse. Was the "
          "recording made with the motion sensor on (--gyro)?")
        return "\n".join(lines) + "\n"
    rate = (f"  {stats['frames']} motion frames over {stats['span_s']:.1f} s "
            f"({stats['rate_hz']:.1f} Hz), {stats['running_frames']} while brushing")
    if stats["running_rate_hz"]:
        rate += f" ({stats['running_rate_hz']:.1f} Hz)"
    w(rate)
    w(f"  {stats['lost']} lost on the way ({stats['lost_pct']:.1f} %)")
    if session.pressure_frames:
        w(f"  {session.pressure_frames} other SensorData frame(s) alongside")

    rows = step_table(session)
    if not rows:
        w("")
        w("No frame could be placed in a pacing step. The recording needs "
          "handle_state, brushing_time and routine_length, and a known mode "
          "(--mode).")
        return "\n".join(lines) + "\n"

    w("")
    w("Steps")
    w("  step zone frames  pitch   roll  spread  motion  quadpacer")
    for r in rows:
        zone = "-" if r["zone"] is None else str(r["zone"])
        quad = ",".join(map(str, r["quadpacer"])) or "-"
        w(f"  {r['step']:>4} {zone:>4} {r['frames']:>6} {r['pitch']:>6.0f}° "
          f"{r['roll']:>5.0f}° {r['spread']:>6.0f}° {r['motion']:>7.0f}  {quad}")

    centres = zone_centres(rows)
    zones = sorted(centres)
    if len(zones) >= 2:
        w("")
        w("Separation (degrees between zone attitudes)")
        w("       " + "".join(f"{z:>6}" for z in zones))
        for a in zones:
            cells = "".join(
                f"{'':>6}" if a == b else f"{angle_deg(centres[a], centres[b]):>6.0f}"
                for b in zones)
            w(f"  {a:>4} {cells}")
        a, b, angle = closest_zones(centres)
        w(f"  closest: zones {a} and {b} ({angle:.0f}°)")

    rec = recognition(rows)
    if rec:
        overall, per_zone = rec
        w("")
        w(f"Recognition (attitude only, same session): {overall:.0f} %")
        w("  " + "  ".join(f"zone {z}: {p:.0f} %" for z, p in per_zone.items()))

    jaws = jaw_coverage(session)
    if jaws and jaws["match_pct"] is not None:
        w("")
        w("Jaws (fixed rule, one second at a time)")
        w(f"  upper {jaws['upper_s']:.0f} s of {jaws['upper_expected_s']:.0f} s, "
          f"lower {jaws['lower_s']:.0f} s of {jaws['lower_expected_s']:.0f} s")
        w(f"  on the paced jaw: {jaws['match_pct']:.0f} %")
        w("  " + "  ".join(f"step {s['step']}: {s['on_jaw_pct']:.0f} %"
                           for s in jaws["steps"]))
    return "\n".join(lines) + "\n"

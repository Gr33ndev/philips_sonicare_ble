"""The motion analysis, and the script that runs it outside Home Assistant."""

from __future__ import annotations

import importlib.util
import json
import random
import struct
import sys
from pathlib import Path

import pytest

from custom_components.philips_sonicare_ble import motion_analysis as analysis
from custom_components.philips_sonicare_ble.const import (
    CHAR_SENSOR_DATA,
    SENSOR_FRAME_GYROSCOPE,
)

SENSOR_UUID = CHAR_SENSOR_DATA.lower()

# One attitude per zone, far enough apart that every zone is recognisable.
ZONE_ATTITUDES = {
    1: (0.0, 0.7, 0.7),
    2: (0.0, 0.0, 1.0),
    3: (0.0, -0.7, 0.7),
    4: (0.0, -0.7, -0.7),
    5: (0.0, 0.0, -1.0),
    6: (0.0, 0.7, -0.7),
}


def _gyro_hex(counter: int, acc, gyro=(10, -5, 3)) -> str:
    return struct.pack("<HH6h", SENSOR_FRAME_GYROSCOPE,
                       counter & 0xFFFF, *acc, *gyro).hex()


def _write_recording(path: Path, rate_hz: float = 20.0, drop_every: int = 0,
                     seed: int = 1, tilt: float = 0.0) -> None:
    rng = random.Random(seed)
    events = [
        {"t": 0.0, "char": "handle_state", "uuid": "x", "hex": "02", "kind": "notify"},
        {"t": 0.0, "char": "brushing_mode", "uuid": "x", "hex": "00", "kind": "notify"},
        {"t": 0.0, "char": "routine_length", "uuid": "x", "hex": "7800", "kind": "notify"},
    ]
    for second in range(120):
        events.append({"t": float(second), "char": "brushing_time", "uuid": "x",
                       "hex": struct.pack("<H", second).hex(), "kind": "notify"})
    counter = 0
    for i in range(int(120 * rate_hz)):
        t = i / rate_hz
        counter += 1
        if drop_every and i % drop_every == 0:
            continue
        zone = int(t // 20) + 1
        ax, ay, az = ZONE_ATTITUDES[zone]
        acc = tuple(int(1000 * c + rng.gauss(0, 60)) for c in (ax + tilt, ay, az))
        events.append({"t": round(t, 3), "char": "sensor_data", "uuid": SENSOR_UUID,
                       "hex": _gyro_hex(counter, acc), "kind": "notify"})
    # A pressure frame on the same characteristic must not count as motion.
    events.append({"t": 5.0, "char": "sensor_data", "uuid": SENSOR_UUID,
                   "hex": struct.pack("<HHhB", 1, 1, 200, 0).hex(), "kind": "notify"})
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "meta", "model": "HX991X"}) + "\n")
        for event in events:
            fh.write(json.dumps(event) + "\n")


def test_parse_gyro_frame_ignores_other_frame_types():
    assert analysis.parse_gyro_frame(_gyro_hex(7, (1, 2, 3), (4, 5, 6))) == (7, (1, 2, 3), (4, 5, 6))
    assert analysis.parse_gyro_frame(struct.pack("<HHhB", 1, 1, 200, 0).hex()) is None


def test_steps_follow_the_pacer(tmp_path):
    path = tmp_path / "rec.jsonl"
    _write_recording(path)
    session = analysis.load_session(path)

    assert session.mode == "clean"
    assert session.pressure_frames == 1
    rows = analysis.step_table(session)
    assert [r["zone"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert all(abs(r["frames"] - 400) <= 2 for r in rows)
    # Zone 2 is held flat: gravity straight down the z axis.
    assert analysis.angle_deg(rows[1]["gravity"], (0.0, 0.0, 1.0)) < 3


def test_a_mode_named_in_the_meta_wins_over_the_raw_value(tmp_path):
    path = tmp_path / "rec.jsonl"
    _write_recording(path)
    meta, events = analysis.read_recording(path)
    session = analysis.session_from_events({**meta, "mode": "white_plus"}, events)
    assert session.mode == "white_plus"
    assert len(analysis.step_table(session)) == 8


def test_stream_stats_count_lost_frames(tmp_path):
    path = tmp_path / "rec.jsonl"
    _write_recording(path, drop_every=10)
    stats = analysis.stream_stats(analysis.load_session(path))

    assert stats["lost"] == pytest.approx(240, abs=1)
    assert stats["lost_pct"] == pytest.approx(10, abs=0.5)
    assert stats["rate_hz"] == pytest.approx(18, abs=0.5)


def test_well_separated_zones_are_recognised(tmp_path):
    path = tmp_path / "rec.jsonl"
    _write_recording(path)
    rows = analysis.step_table(analysis.load_session(path))

    overall, per_zone = analysis.recognition(rows)
    assert overall > 95
    assert set(per_zone) == set(ZONE_ATTITUDES)
    centres = analysis.zone_centres(rows)
    assert analysis.angle_deg(centres[1], centres[2]) == pytest.approx(45, abs=3)


def test_recognition_across_sessions(tmp_path):
    sessions = []
    for seed in (1, 2, 3):
        path = tmp_path / f"rec{seed}.jsonl"
        _write_recording(path, seed=seed)
        sessions.append(analysis.load_session(path))
    overall, per_session = analysis.cross_session_recognition(sessions)
    assert overall > 95
    assert len(per_session) == 3
    assert analysis.cross_session_recognition(sessions[:1]) is None


def test_summary_is_json_safe_and_small(tmp_path):
    path = tmp_path / "rec.jsonl"
    _write_recording(path)
    result = analysis.summary(analysis.load_session(path))
    encoded = json.dumps(result)
    # Kept well under the size the recorder stores attributes up to.
    assert len(encoded) < 4000
    assert result["closest_angle"] == 45
    assert set(result["zone_attitudes"]) == {"1", "2", "3", "4", "5", "6"}


def test_recording_without_motion_says_so(tmp_path):
    path = tmp_path / "rec.jsonl"
    path.write_text(json.dumps({"kind": "meta", "model": "HX991X"}) + "\n")
    session = analysis.load_session(path)
    assert "--gyro" in analysis.format_report(session)
    assert analysis.summary(session)["frames"] == 0


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "sonicare_gyro_analyze.py"
    spec = importlib.util.spec_from_file_location("sonicare_gyro_analyze", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_reports_several_recordings(tmp_path, monkeypatch, capsys):
    script = _load_script()
    paths = []
    for seed in (1, 2):
        path = tmp_path / f"rec{seed}.jsonl"
        _write_recording(path, seed=seed)
        paths.append(str(path))
    csv_path = tmp_path / "frames.csv"
    monkeypatch.setattr(sys, "argv", ["x", *paths, "--csv", str(csv_path)])
    script.main()
    out = capsys.readouterr().out
    assert "Separation" in out
    assert "Recognition across sessions" in out
    assert len(csv_path.read_text().splitlines()) == 2401

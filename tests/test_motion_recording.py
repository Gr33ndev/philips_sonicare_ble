"""Recording a session's motion data inside Home Assistant.

The point of the recording is to find out whether the motion sensor can tell
the zones of the mouth apart, so what matters is that a session comes out of
it whole: every frame that arrived, the values needed to place them, and the
bridge throttle out of the way while it runs and back afterwards.
"""
from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from types import SimpleNamespace

from custom_components.philips_sonicare_ble import motion_analysis
from custom_components.philips_sonicare_ble.const import (
    CHAR_BRUSHING_TIME,
    CHAR_SENSOR_DATA,
    CONF_NOTIFY_THROTTLE,
    SENSOR_ENABLE_GYROSCOPE,
    SENSOR_ENABLE_PRESSURE,
    SENSOR_FRAME_GYROSCOPE,
)
from custom_components.philips_sonicare_ble.coordinator import (
    PhilipsSonicareCoordinator,
)
from custom_components.philips_sonicare_ble.motion_recorder import (
    MotionRecorder,
    write_recording,
)

ZONE_ATTITUDES = {
    1: (0, 700, 700),
    2: (0, 0, 1000),
    3: (0, -700, 700),
    4: (0, -700, -700),
    5: (0, 0, -1000),
    6: (0, 700, -700),
}


def _gyro(counter: int, acc) -> bytes:
    return struct.pack("<HH6h", SENSOR_FRAME_GYROSCOPE, counter, *acc, 10, -5, 3)


class _Transport:
    is_connected = True

    def __init__(self) -> None:
        self.throttles: list[int] = []

    async def set_notify_throttle(self, ms: int) -> None:
        self.throttles.append(ms)


class _Protocol:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    async def start_sensor_stream(self, mask, cb) -> bool:
        return self.ok

    async def stop_sensor_stream(self) -> None:
        pass


def _coordinator(tmp_path: Path, gyro: bool = True, esp: bool = True, ok: bool = True):
    c = PhilipsSonicareCoordinator.__new__(PhilipsSonicareCoordinator)
    c.address = "AA:BB"
    c._use_condor = False
    c._is_esp_bridge = esp
    c._sensor_subscribed = False
    c._motion_recorder = None
    c._live_cb = lambda *a: None
    c.transport = _Transport()
    c._protocol = _Protocol(ok)
    c.entry = SimpleNamespace(
        entry_id="entry1",
        data={"model": "HX992X"},
        options={"sensor_gyroscope": gyro, "sensor_pressure": True,
                 CONF_NOTIFY_THROTTLE: 500},
    )
    c.data = {
        "model_number": "HX992X",
        "brushing_mode": "clean",
        "routine_length": 120,
        "brushing_time": 0,
        "handle_state_value": 2,
    }
    c.tasks = []
    c.updates = []

    async def _executor(func, *args):
        return func(*args)

    c.hass = SimpleNamespace(
        config=SimpleNamespace(path=lambda *p: str(tmp_path.joinpath(*p))),
        async_create_task=lambda coro: c.tasks.append(coro),
        async_add_executor_job=_executor,
    )
    c.async_set_updated_data = lambda data: c.updates.append(data)
    return c


def _brush(c, seconds: int = 120, rate: int = 10) -> None:
    """Feed a routine the way the live callback would."""
    counter = 0
    recorder = c._motion_recorder
    for i in range(seconds * rate):
        t = i / rate
        if i % rate == 0:
            recorder.record(CHAR_BRUSHING_TIME, int(t).to_bytes(2, "little"))
            recorder.events[-1]["t"] = t
        counter += 1
        zone = int(t // 20) + 1
        recorder.record(CHAR_SENSOR_DATA, _gyro(counter, ZONE_ATTITUDES[zone]))
        recorder.events[-1]["t"] = t


async def test_a_session_is_recorded_analysed_and_the_throttle_restored(tmp_path):
    c = _coordinator(tmp_path)
    await c._subscribe_sensor_data_locked()

    assert c._sensor_subscribed
    assert c._motion_recorder is not None
    assert c.transport.throttles == [0], "lifted for the session"

    _brush(c)
    await c._unsubscribe_sensor_data_locked()
    assert c._motion_recorder is None
    assert c.transport.throttles == [0, 500], "back to the configured value"

    assert len(c.tasks) == 1
    await c.tasks[0]

    result = c.updates[-1]["motion_recording"]
    assert result["frames"] == 1200
    assert result["mode"] == "clean"
    assert [s["zone"] for s in result["steps"]] == [1, 2, 3, 4, 5, 6]
    assert result["recognition_pct"] == 100
    assert result["ended"] == "session ended"

    path = tmp_path / result["file"]
    meta, events = motion_analysis.read_recording(path)
    assert meta["model"] == "HX992X"
    assert "address" not in json.dumps(meta).lower()
    assert sum(e["char"] == "sensor_data" for e in events) == 1200
    # The recording reads back into the same analysis.
    again = motion_analysis.summary(motion_analysis.load_session(path))
    assert again["steps"] == result["steps"]


async def test_nothing_is_recorded_without_the_motion_sensor(tmp_path):
    c = _coordinator(tmp_path, gyro=False)
    await c._subscribe_sensor_data_locked()
    assert c._sensor_subscribed
    assert c._motion_recorder is None
    assert c.transport.throttles == [], "the bridge keeps its throttle"
    await c._unsubscribe_sensor_data_locked()
    assert c.tasks == []


async def test_a_direct_connection_has_no_throttle_to_lift(tmp_path):
    c = _coordinator(tmp_path, esp=False)
    await c._subscribe_sensor_data_locked()
    assert c._motion_recorder is not None
    assert c.transport.throttles == []


async def test_a_failed_subscribe_gives_the_throttle_back(tmp_path):
    c = _coordinator(tmp_path, ok=False)
    await c._subscribe_sensor_data_locked()
    assert not c._sensor_subscribed
    assert c._motion_recorder is None
    assert c.transport.throttles == [0, 500]


async def test_a_dropped_link_still_keeps_the_recording(tmp_path):
    c = _coordinator(tmp_path)
    await c._subscribe_sensor_data_locked()
    _brush(c, seconds=30)
    assert c._finish_motion_recording("link dropped")
    await c.tasks[0]
    assert c.updates[-1]["motion_recording"]["ended"] == "link dropped"
    assert not c._finish_motion_recording("link dropped"), "only once"


def test_the_baseline_places_frames_that_arrive_before_any_change(tmp_path):
    """The handle reports its state once, before the recording starts.

    Without the baseline the analysis would never see the handle running and
    would place no frame at all.
    """
    c = _coordinator(tmp_path)
    asyncio.run(c._start_motion_recording())
    names = [e["char"] for e in c._motion_recorder.events]
    assert names == ["handle_state", "brushing_time", "routine_length"]


def test_old_recordings_are_pruned(tmp_path):
    for i in range(5):
        write_recording(tmp_path / f"sonicare_session_hx992x_2026010{i}_000000.jsonl",
                        {"kind": "meta"}, [], keep=3)
    kept = sorted(p.name for p in tmp_path.iterdir())
    assert kept == [f"sonicare_session_hx992x_2026010{i}_000000.jsonl" for i in (2, 3, 4)]


def test_recorder_stops_growing_at_its_limit(monkeypatch):
    from custom_components.philips_sonicare_ble import motion_recorder
    monkeypatch.setattr(motion_recorder, "MAX_EVENTS", 3)
    r = MotionRecorder({})
    for _ in range(5):
        r.record(CHAR_SENSOR_DATA, b"\x01")
    assert len(r.events) == 3
    assert r.finish("session ended")["truncated"] is True


def test_the_mask_alone_decides_whether_to_record(tmp_path):
    c = _coordinator(tmp_path)
    assert c._motion_recording_wanted(SENSOR_ENABLE_GYROSCOPE)
    assert not c._motion_recording_wanted(SENSOR_ENABLE_PRESSURE)
    c._use_condor = True
    assert not c._motion_recording_wanted(SENSOR_ENABLE_GYROSCOPE)

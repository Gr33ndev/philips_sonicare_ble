"""Record a brushing session's motion data inside Home Assistant.

With the motion sensor switched on in the options, every Classic session is
kept as a recording: each characteristic value that arrives while the brush
runs, timestamped, in the JSONL shape ``scripts/sonicare_session_record.py``
writes. A recording made here can therefore be read by the same analysis, on
the Home Assistant host or anywhere else, without taking the brush away from
Home Assistant to record it with the script.

What it is for is finding out whether the motion sensor can tell the zones of
the mouth apart at all (see ``motion_analysis.py``). That needs sessions, and
sessions are what Home Assistant sees every day anyway.

The recordings carry the model and nothing that identifies the brush or the
person - no address, no serial - like the script's.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from . import const

_LOGGER = logging.getLogger(__name__)

# uuid -> short name, derived from the CHAR_* constants the same way the
# recording script does it (CHAR_BRUSHING_TIME becomes "brushing_time"), so
# both kinds of recording name their values alike.
CHAR_NAMES = {
    value.lower(): name[len("CHAR_"):].lower()
    for name, value in vars(const).items()
    if name.startswith("CHAR_") and isinstance(value, str)
}

RECORDINGS_DIR = "philips_sonicare_ble/recordings"

# Recordings kept per brush; older ones are deleted when a new one is
# written. A two-minute session at the rates a Classic handle sends is around
# a megabyte, so this bounds the disk use to a few dozen.
KEEP_RECORDINGS = 30

# A session that never ends must not grow without limit. Far above anything
# a routine produces - a routine is two minutes, this is minutes of frames
# at a rate no handle has been seen to reach.
MAX_EVENTS = 200_000


class MotionRecorder:
    """Collects the values of one session, in arrival order."""

    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.events: list[dict[str, Any]] = []
        self.started = time.monotonic()
        self.wall_started = time.time()
        self.truncated = False

    def record(self, uuid: str, data: bytes, kind: str = "notify") -> None:
        if len(self.events) >= MAX_EVENTS:
            self.truncated = True
            return
        key = uuid.lower()
        self.events.append({
            "t": round(time.monotonic() - self.started, 3),
            "kind": kind,
            "char": CHAR_NAMES.get(key, key[4:8]),
            "uuid": key,
            "hex": bytes(data).hex(),
        })

    def baseline(self, uuid: str, data: bytes) -> None:
        """A value the session started with, recorded at t=0.

        A characteristic notifies on change, so what the handle was already
        reporting when the recording started never arrives as a notification.
        The analysis needs some of it - the handle state and the routine
        length above all - to place the frames at all.
        """
        self.record(uuid, data, "baseline")

    def finish(self, ended: str) -> dict[str, Any]:
        """The meta line for the recording, completed now that it is over."""
        meta = dict(self.meta)
        meta.update({
            "kind": "meta",
            "protocol": "classic",
            "source": "home_assistant",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.wall_started)),
            "duration_s": round(time.monotonic() - self.started, 1),
            "events": len(self.events),
            "ended": ended,
            "truncated": self.truncated,
        })
        return meta


def recording_path(base: Path, device_id: str, model: str | None, started: float) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(started))
    name = f"sonicare_session_{(model or 'unknown').lower()}_{stamp}.jsonl"
    return base / device_id / name


def write_recording(path: Path, meta: dict[str, Any], events: list[dict[str, Any]],
                    keep: int = KEEP_RECORDINGS) -> None:
    """Write one recording and drop the oldest beyond ``keep``. Blocking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    existing = sorted(path.parent.glob("sonicare_session_*.jsonl"))
    for old in existing[:-keep] if keep > 0 else []:
        try:
            old.unlink()
        except OSError as err:
            _LOGGER.debug("Could not remove old recording %s: %s", old, err)

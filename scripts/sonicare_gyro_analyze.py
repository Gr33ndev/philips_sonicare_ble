#!/usr/bin/env python3
"""Look at the motion data in Sonicare session recordings.

The question this answers is whether the handle's own motion sensor can tell
the zones of the mouth apart - which is what a position-aware app does, and
what the integration cannot do yet. The measures themselves live in the
integration (``motion_analysis.py``), so a recording is judged the same way
here as in Home Assistant:

- **Stream** - how many motion frames arrived and how many were lost on the
  way. A position estimate is only as good as the link that carries it.
- **Steps** - one row per pacing step: the zone, the handle's average
  attitude, how steady it was held, and how much it was moved.
- **Separation** - the angle between the average attitudes of every two
  zones. A few degrees means the motion sensor cannot tell them apart.
- **Recognition** - learned from each step's first half, tested on its
  second. Optimistic, because both come from the same session.
- **Across sessions** - with two or more recordings, each one is classified
  against the zones learned from all the others. That is the figure that says
  whether this works on the next session.

Recordings come from the integration (with the motion sensor switched on in
its options they land in ``<config>/philips_sonicare_ble/recordings/``) or
from ``sonicare_session_record.py --gyro``.

Usage:
  python sonicare_gyro_analyze.py session.jsonl
  python sonicare_gyro_analyze.py recordings/*.jsonl
  python sonicare_gyro_analyze.py session.jsonl --csv frames.csv

Brush the routine as the handle paces it, zone by zone, without starting
early or lingering. Anything else makes the steps wrong labels, and the
figures with them.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import types
from pathlib import Path

_PACKAGE_DIR = (Path(__file__).resolve().parent.parent
                / "custom_components" / "philips_sonicare_ble")


def _load_analysis():
    """Import motion_analysis without the package's Home Assistant imports.

    The package __init__ pulls in Home Assistant, and this script has to run
    on a bare Python. Registering an empty package under the same directory
    lets the module's relative import of const resolve, and skips __init__.
    """
    package = types.ModuleType("_sonicare_pkg")
    package.__path__ = [str(_PACKAGE_DIR)]
    sys.modules.setdefault("_sonicare_pkg", package)
    return importlib.import_module("_sonicare_pkg.motion_analysis")


analysis = _load_analysis()


def write_csv(session, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "counter", "elapsed", "step", "zone",
                         "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        for f in session.frames:
            writer.writerow([
                f.t, f.counter,
                "" if f.elapsed is None else round(f.elapsed, 3),
                "" if f.step is None else f.step + 1,
                "" if f.zone is None else f.zone,
                *f.acc, *f.gyro,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse the motion stream in Sonicare session recordings")
    parser.add_argument("recordings", type=Path, nargs="+",
                        help="JSONL recordings (integration or sonicare_session_record.py)")
    parser.add_argument("--mode", help="Brushing routine, when a recording "
                                       "does not say (e.g. clean)")
    parser.add_argument("--csv", type=Path,
                        help="Also write every motion frame of the first recording to this CSV")
    args = parser.parse_args()

    sessions = []
    for path in args.recordings:
        session = analysis.load_session(path, args.mode)
        sessions.append(session)
        if len(args.recordings) > 1:
            print(f"=== {path.name}")
        print(analysis.format_report(session))

    cross = analysis.cross_session_recognition(sessions)
    if cross:
        overall, per_session = cross
        print(f"Recognition across sessions (learned from the others): {overall:.0f} %")
        print("  " + "  ".join(f"{p:.0f} %" for p in per_session))

    if args.csv:
        write_csv(sessions[0], args.csv)
        print(f"\n{len(sessions[0].frames)} frame(s) written to {args.csv}")


if __name__ == "__main__":
    main()

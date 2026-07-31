#!/usr/bin/env python3
"""Run cache_walk and add Linux perf counters when they are available."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess


EVENTS = ("cycles", "instructions", "cache-misses", "branch-misses")


def run(command: list[str]) -> tuple[list[str], dict[str, float], bool]:
    counters = {event: 0.0 for event in EVENTS}
    perf = shutil.which("perf") if platform.system() == "Linux" else None
    if perf:
        completed = subprocess.run(
            [perf, "stat", "-x,", "-e", ",".join(EVENTS), "--", *command],
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0:
            for line in completed.stderr.splitlines():
                fields = line.split(",")
                if len(fields) >= 3 and fields[2] in counters:
                    try:
                        counters[fields[2]] = float(fields[0])
                    except ValueError:
                        pass
            return completed.stdout.split(), counters, any(counters.values())

    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or f"cache_walk exited {completed.returncode}")
    return completed.stdout.split(), counters, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--mib", type=int, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    values, counters, perf_available = run([
        args.program,
        str(args.mib),
        args.pattern,
        str(args.stride),
        str(args.seed),
    ])
    if len(values) != 3:
        raise SystemExit(f"unexpected cache_walk output: {values!r}")
    print(*values, *(counters[event] for event in EVENTS), int(perf_available))


if __name__ == "__main__":
    main()

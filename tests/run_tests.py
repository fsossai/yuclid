#!/usr/bin/env python3
"""Self-contained test suite for `yuclid run`.

Every case lives in `cases/<name>/`. The directory holds the configuration
file(s) the case needs plus a `case.json` describing how to invoke yuclid and
what the run must produce. Cases are hermetic: the directory is copied into a
fresh temporary working directory, so a case never sees the repository, the
user's files or another case's leftovers, and every trial is a deterministic
shell command whose output is fixed.

Cases are independent, so they run concurrently; use -j1 to serialize them.

Usage:
    python tests/run_tests.py [-k PATTERN] [-j N] [-v] [--keep] [--list]
"""

import argparse
import concurrent.futures
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CASES_DIR = os.path.join(HERE, "cases")
RESULTS = "results.jsonl"


class Failure(Exception):
    pass


def load_case(path):
    with open(os.path.join(path, "case.json")) as f:
        return json.load(f)


def normalize(value):
    """Make a parsed record comparable: NaN has no equality, so name it."""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    return value


def same(actual, expected):
    """Equality that refuses to confuse an integer with an equal float."""
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            same(actual[k], expected[k]) for k in expected
        )
    return actual == expected


def key(record):
    return json.dumps(record, sort_keys=True)


def read_records(workdir, case):
    path = os.path.join(workdir, RESULTS)
    if not os.path.isfile(path):
        return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(normalize(json.loads(line)))
    return records


def child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # keep the diagnostics reproducible and independent of the caller's locale
    env["LC_ALL"] = "C"
    env.pop("PYTHONWARNINGS", None)
    return env


def find_python(explicit):
    """Pick an interpreter that can import the working copy of yuclid.

    `yuclid.cli` imports the plotting stack eagerly, so `run` cannot be
    exercised by an interpreter that only has pandas.
    """
    candidates = [explicit] if explicit else []
    candidates += [os.environ.get("YUCLID_TEST_PYTHON"), sys.executable]
    candidates += ["python3.12", "python3.11", "python3.13", "python3"]
    tried = []
    for candidate in candidates:
        if not candidate or candidate in tried:
            continue
        tried.append(candidate)
        try:
            proc = subprocess.run(
                [candidate, "-c", "from yuclid.cli import main"],
                env=child_env(),
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        if proc.returncode == 0:
            return candidate
    sys.exit(
        "no interpreter can import yuclid; tried: {}\n"
        "install the dependencies or set YUCLID_TEST_PYTHON".format(", ".join(tried))
    )


PYTHON = sys.executable

# the console script installed by pyproject.toml, reproduced verbatim.
# `python -m yuclid.cli` is *not* equivalent: it dies on a circular import
# between yuclid.cli, yuclid.plot and yuclid.tplot.
ENTRY_POINT = "import sys; from yuclid.cli import main; sys.exit(main())"


def invoke(argv, workdir, timeout):
    env = child_env()
    try:
        proc = subprocess.run(
            [PYTHON, "-c", ENTRY_POINT] + argv,
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise Failure("timed out after {}s: yuclid {}".format(timeout, " ".join(argv)))
    return proc.returncode, proc.stdout + proc.stderr


def check(case, workdir, codes, output):
    expected_codes = case.get("exit_codes")
    if expected_codes is None:
        expected_codes = [0] * (len(codes) - 1) + [case.get("exit_code", 0)]
    if codes != expected_codes:
        raise Failure("exit code {}, expected {}".format(codes, expected_codes))

    for needle in case.get("stdout_contains", []):
        if needle not in output:
            raise Failure("output does not contain {!r}".format(needle))
    for needle in case.get("stdout_not_contains", []):
        if needle in output:
            raise Failure("output unexpectedly contains {!r}".format(needle))

    for pattern in case.get("files_exist", []):
        if not glob.glob(os.path.join(workdir, pattern)):
            raise Failure("no file matches {!r}".format(pattern))
    for pattern in case.get("files_not_exist", []):
        if glob.glob(os.path.join(workdir, pattern)):
            raise Failure("a file unexpectedly matches {!r}".format(pattern))

    records = read_records(workdir, case)

    if case.get("no_results_file"):
        if records is not None:
            raise Failure("a result dataset was written but none was expected")
        return
    wants_records = any(
        k in case for k in ("records", "records_unordered", "record_count")
    )
    if not wants_records:
        return
    if records is None:
        raise Failure("no result dataset was written")

    if "record_count" in case:
        if len(records) != case["record_count"]:
            raise Failure(
                "{} records, expected {}".format(len(records), case["record_count"])
            )

    if "records" in case:
        expected = [normalize(r) for r in case["records"]]
        if len(records) != len(expected) or not all(
            same(a, e) for a, e in zip(records, expected)
        ):
            raise Failure(diff(records, expected, "in order"))

    if "records_unordered" in case:
        expected = [normalize(r) for r in case["records_unordered"]]
        got = sorted(records, key=key)
        want = sorted(expected, key=key)
        if len(got) != len(want) or not all(same(a, e) for a, e in zip(got, want)):
            raise Failure(diff(got, want, "as a set"))


def diff(actual, expected, how):
    lines = ["records differ ({})".format(how), "  expected:"]
    lines += ["    " + key(r) for r in expected] or ["    (none)"]
    lines.append("  actual:")
    lines += ["    " + key(r) for r in actual] or ["    (none)"]
    return "\n".join(lines)


def run_case(name, keep, timeout):
    src = os.path.join(CASES_DIR, name)
    case = load_case(src)
    workdir = tempfile.mkdtemp(prefix="yuclid-test-{}-".format(name))
    try:
        for entry in os.listdir(src):
            if entry == "case.json":
                continue
            shutil.copy2(os.path.join(src, entry), os.path.join(workdir, entry))

        runs = case.get("runs") or [case.get("args", ["run"])]
        codes, output = [], ""
        try:
            for argv in runs:
                argv = list(argv)
                if case.get("output_flag", True):
                    argv += ["-o", RESULTS]
                code, out = invoke(argv, workdir, timeout)
                codes.append(code)
                output += out
            check(case, workdir, codes, output)
        except Failure as e:
            return "fail", str(e), output, workdir
        return "pass", "", output, workdir
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description="Run the `yuclid run` test suite")
    p.add_argument("-k", metavar="PATTERN", help="only run cases whose name contains this")
    p.add_argument("-v", "--verbose", action="store_true", help="show yuclid's output")
    p.add_argument("--keep", action="store_true", help="keep the temporary working directories")
    p.add_argument("--list", action="store_true", help="list the cases and what they cover")
    p.add_argument("--python", help="interpreter to run yuclid with (default: autodetected)")
    p.add_argument(
        "-j", "--jobs", type=int, default=min(16, (os.cpu_count() or 4)),
        help="how many cases to run at once (default: %(default)s)",
    )
    p.add_argument(
        "--timeout", type=float, default=60.0,
        help="seconds a single yuclid invocation may take (default: %(default)s)",
    )
    args = p.parse_args()

    names = sorted(os.listdir(CASES_DIR))
    names = [n for n in names if os.path.isdir(os.path.join(CASES_DIR, n))]
    if args.k:
        names = [n for n in names if args.k in n]

    if args.list:
        for name in names:
            case = load_case(os.path.join(CASES_DIR, name))
            print(name)
            print("    {}".format(case["description"]))
        return 0

    global PYTHON
    PYTHON = find_python(args.python)

    tally = {"pass": 0, "fail": 0, "xfail": 0, "xpass": 0}
    failures = []

    # a case costs about a second and a half, nearly all of it spent importing
    # the plotting stack that `yuclid.cli` pulls in, so run them concurrently:
    # every case owns its working directory and shares nothing with the others.
    # each one is reported the moment it lands, so the order below is the order
    # they finished in, not the order they were listed in.
    started = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {
            pool.submit(run_case, name, args.keep, args.timeout): name
            for name in names
        }
        for future in concurrent.futures.as_completed(pending):
            name = pending[future]
            outcome, reason, output, workdir = future.result()
            case = load_case(os.path.join(CASES_DIR, name))
            xfail = case.get("xfail")
            if xfail:
                outcome = "xfail" if outcome == "fail" else "xpass"
            tally[outcome] += 1
            done += 1

            mark = {
                "pass": "PASS", "fail": "FAIL", "xfail": "XFAIL", "xpass": "XPASS"
            }[outcome]
            print("[{:>{w}}/{}] {:<5} {}".format(
                done, len(names), mark, name, w=len(str(len(names)))
            ), flush=True)
            indent = " " * (len(str(len(names))) * 2 + 4)
            if outcome == "xfail":
                print(indent + "known defect: {}".format(xfail))
            if outcome == "xpass":
                print(indent + "the defect seems fixed — drop 'xfail' from case.json")
                print(indent + "it was: {}".format(xfail))
            if outcome == "fail":
                for line in reason.splitlines():
                    print(indent + line)
                failures.append(name)
            if args.keep:
                print(indent + "workdir: {}".format(workdir))
            if args.verbose and outcome in ("fail", "xfail"):
                for line in output.splitlines():
                    print(indent + "| " + line)

    print(
        "\nran in {:.1f}s".format(time.time() - started)
    )
    print(
        "{pass} passed, {fail} failed, {xfail} known defects, {xpass} unexpectedly fixed".format(
            **tally
        )
    )
    if failures:
        print("failed: " + ", ".join(sorted(failures)))
    return 1 if tally["fail"] or tally["xpass"] else 0


if __name__ == "__main__":
    sys.exit(main())

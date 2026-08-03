"""The commands that look at runs and steer them.

Looking reads the run's files, which works the same whether it is still going or
finished long ago. Steering calls the run over its socket, which only a live run
answers.
"""

from yuclid.log import LogLevel, report
import yuclid.workspace as workspace
import yuclid.control as control
import os
import time


def find_root_or_fail():
    root = workspace.find_root()
    if root is None:
        report(
            LogLevel.FATAL,
            "no .yuclid directory here",
            hint="`yuclid run` creates one; `mkdir .yuclid` pins it to a "
            "directory of your choosing",
        )
    return root


def progress_counts(directory):
    """How far a run got, as (completed, total), or None when it never said.

    The total is taken from the last record that carries one rather than from
    the plan the run started with: dropping and adding points move it, and the
    figure that means anything is the one that was true most recently.
    """
    total, done = None, 0
    for record in workspace.read_progress(directory):
        if record.get("total") is not None:
            total = record["total"]
        if record["type"] == "point.finished":
            done += record.get("repetitions", 1)
    if total is None:
        return None
    return done, total


def format_counts(directory):
    counts = progress_counts(directory)
    if counts is None:
        return "-"
    return "{}/{}".format(*counts)


def launch_runs(args):
    root = find_root_or_fail()
    runs = workspace.list_runs(root)

    if args.last:
        if len(runs) == 0:
            report(LogLevel.FATAL, "no run has been recorded yet")
        print(runs[0].get("output") or "")
        return

    if len(runs) == 0:
        report(
            LogLevel.INFO,
            "no run has been recorded yet",
            hint="`yuclid run` records every run it makes",
        )
        return

    rows = []
    for manifest in runs[: args.n]:
        rows.append(
            (
                manifest["id"],
                manifest["state"],
                format_counts(manifest["directory"]),
                relative(manifest.get("output")),
            )
        )

    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    for row in rows:
        print(
            "  ".join(
                [
                    row[0].ljust(widths[0]),
                    row[1].ljust(widths[1]),
                    row[2].rjust(widths[2]),
                    row[3],
                ]
            )
        )


def relative(path):
    """A path the user can paste, preferring the short form when there is one."""
    if not path:
        return ""
    try:
        short = os.path.relpath(path)
    except ValueError:
        return path
    return short if len(short) < len(path) else path


def call(args, operation):
    """Send one operation to the run it is addressed to, and say what it did."""
    root = find_root_or_fail()
    manifest = workspace.select_run(root, getattr(args, "run", None))
    if manifest["state"] != workspace.RUNNING:
        report(
            LogLevel.FATAL,
            "run {} is {}".format(manifest["id"], manifest["state"]),
            hint="only a run in progress can be steered",
        )
    path = os.path.join(manifest["directory"], workspace.CONTROL)
    try:
        return control.request(path, operation)
    except control.ControlError as e:
        report(LogLevel.FATAL, str(e))
    except OSError as e:
        report(
            LogLevel.FATAL,
            "cannot reach run {}".format(manifest["id"]),
            str(e),
            hint="a run can only be steered from the machine it runs on",
        )


def launch_operation(args):
    """Every steering command: build the operation, send it, report the effect."""
    operation = {"op": args.command}

    if args.command == "kill":
        operation["scope"] = args.scope
        if args.point:
            operation["coords"] = _coords(args.point)
    elif args.command in ("drop", "add"):
        operation["coords"] = _coords(args.coords)
    elif args.command == "repeat":
        operation["value"] = args.value
    elif args.command == "order":
        operation["order"] = args.dimensions

    effect = call(args, operation)
    report(LogLevel.INFO, args.command, describe(args.command, effect))


def _coords(pairs):
    from yuclid.run import parse_coordinates

    return parse_coordinates(pairs, what="coordinate")


def describe(command, effect):
    """What happened, in units the person who typed the command thinks in."""
    remaining = effect.get("total", 0) - effect.get("completed", 0)
    if command == "drop":
        return "{} point(s) dropped, {} unit(s) remain".format(
            effect.get("dropped", 0), remaining
        )
    if command == "add":
        return "{} point(s) added, {} restored, {} unit(s) remain".format(
            effect.get("added", 0), effect.get("restored", 0), remaining
        )
    if command == "kill":
        points = ", ".join(".".join(p) for p in effect.get("points", []))
        return "{} abandoned on {}".format(
            "the repetition" if effect.get("scope") == "rep" else "the point", points
        )
    if command == "pause":
        return "paused with {} point(s) still in flight".format(
            effect.get("in_flight", 0)
        )
    if command == "resume":
        return "resumed, {} unit(s) remain".format(remaining)
    if command == "stop":
        return "stopping, {} point(s) abandoned".format(effect.get("abandoned", 0))
    if command == "repeat":
        return "{} point(s) now ask for {} repetition(s)".format(
            effect.get("retargeted", 0), effect.get("repeat")
        )
    if command == "order":
        return "{} pending point(s) reordered as {}".format(
            effect.get("reordered", 0), " ".join(effect.get("order", []))
        )
    return str(effect)


def launch_status(args):
    root = find_root_or_fail()
    manifest = workspace.select_run(root, getattr(args, "run", None))
    while True:
        line = status_line(manifest)
        if not args.watch:
            print(line)
            return
        print("\r\033[K" + line, end="", flush=True)
        if workspace.state_of(workspace.read_manifest(manifest["directory"])) != (
            workspace.RUNNING
        ):
            print()
            return
        time.sleep(1.0)


def status_line(manifest):
    directory = manifest["directory"]
    state = workspace.state_of(workspace.read_manifest(directory) or manifest)
    counts = progress_counts(directory)
    where = ""
    for record in reversed(workspace.read_progress(directory)):
        if record["type"] == "point.started":
            where = " " + ".".join(record["key"])
            break
    return "{}  {}  {}{}".format(
        manifest["id"],
        state,
        "[{}/{}]".format(*counts) if counts else "-",
        where,
    )

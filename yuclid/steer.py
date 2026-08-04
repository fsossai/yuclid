"""The commands that look at runs and steer them.

Looking reads the run's files, which works the same whether it is still going or
finished long ago. Steering calls the run over its socket, which only a live run
answers.
"""

from yuclid.log import LogLevel, report
import yuclid.workspace as workspace
import yuclid.control as control
from datetime import datetime
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
        if record["type"] in ("point.finished", "point.skipped"):
            done += record.get("repetitions", 1)
    if total is None:
        return None
    return done, total


def format_counts(directory):
    """The same counter the run prints, so one format is learnt once."""
    from yuclid.run import get_progress

    counts = progress_counts(directory)
    if counts is None:
        return "-"
    return get_progress(*counts)


def failed_points(directory):
    """The points that ran but recorded nothing, so the gaps have a cause."""
    failed, plan = set(), None
    for record in workspace.read_progress(directory):
        if record["type"] == "plan":
            plan = record
        if record["type"] == "point.finished" and record.get("failed"):
            failed.add(tuple(record["key"]))
    if plan is not None:
        failed |= {
            tuple(p["key"]) for p in plan["points"] if p["status"] == "failed"
        }
    return sorted(failed)


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
        failed = failed_points(manifest["directory"])
        state = manifest["state"]
        if failed:
            state += " ({} failed)".format(len(failed))
        rows.append(
            (
                manifest["id"],
                manifest.get("name") or "",
                state,
                format_counts(manifest["directory"]),
                relative(manifest.get("output")),
            )
        )

    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for row in rows:
        print(
            "  ".join(
                [
                    row[0].ljust(widths[0]),
                    row[1].ljust(widths[1]),
                    row[2].ljust(widths[2]),
                    row[3].rjust(widths[3]),
                    row[4],
                ]
            ).rstrip()
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

    # steering names parts of a space, and a dimension on its own is one
    return parse_coordinates(pairs, what="coordinate", whole=True)


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
        flight = effect.get("in_flight", 0)
        if flight == 0:
            return "paused: nothing was running"
        return (
            "pausing: no new point will start, and the {} in flight will stop "
            "as soon as the command they are running is done".format(flight)
        )
    if command == "resume":
        return "resumed, {} unit(s) remain".format(remaining)
    if command == "stop":
        return "stopped: {} point(s) abandoned, {} interrupted".format(
            effect.get("abandoned", 0), effect.get("interrupted", 0)
        )
    if command == "repeat":
        return "{} point(s) now ask for {} repetition(s)".format(
            effect.get("retargeted", 0), effect.get("repeat")
        )
    if command == "order":
        return "{} pending point(s) reordered as {}".format(
            effect.get("reordered", 0), " ".join(effect.get("order", []))
        )
    return str(effect)


def source_run(run_id):
    """A previous run's directory and the command that made it."""
    root = find_root_or_fail()
    directory = workspace.run_directory(root, run_id)
    manifest = workspace.read_manifest(directory)
    if manifest is None:
        report(
            LogLevel.FATAL,
            "no such run",
            run_id,
            hint="`yuclid runs` lists what there is",
        )
    argv = list(manifest.get("argv") or [])
    if argv[:1] != ["run"]:
        report(
            LogLevel.FATAL,
            "run {} was not started by `yuclid run`".format(run_id),
        )
    manifest["directory"] = directory
    return manifest, argv


def launch_finish(args, parser):
    """Run again whatever a previous run did not record.

    Points that failed, were killed, or never started left no record behind, so
    resuming against the same file is exactly the work still outstanding — the
    same mechanism that finishes a run cut short by Ctrl-C.
    """
    import yuclid.run

    manifest, argv = source_run(args.run)
    output = manifest.get("output")
    if not output or not os.path.exists(output):
        report(
            LogLevel.FATAL,
            "the results of run {} are gone".format(args.run),
            output or "",
            hint="there is nothing to resume from, so start the run again",
        )

    argv = strip_option(argv, "--replay")
    argv = strip_option(argv, "-o")
    argv = strip_option(argv, "--output")
    argv = strip_option(argv, "--output-dir")
    argv = [x for x in argv if x != "--resume"]
    argv += ["-o", output, "--resume"]

    report(LogLevel.INFO, "resuming", " ".join(argv))
    yuclid.run.launch(as_run(parser, argv))


def launch_replay(args, parser):
    """Run the points a previous run measured, again.

    A replay is that run's result rather than its instructions: whatever it was
    told along the way — a slice dropped, a point killed, the whole thing
    stopped early — the points it came to are the points to do again. Those go
    into a file and the run is an ordinary `--points` run, which is also what
    makes a replay something the caller can edit before starting it.

    The command comes from the source run's manifest, so the trials, the
    repetitions and the environment are the same experiment and not an
    approximation of it. Where it wrote is not part of that: a replay is a run
    of its own and gets a results file of its own.
    """
    import yuclid.run

    manifest, argv = source_run(args.run)
    argv = strip_option(argv, "--replay")
    argv = strip_option(argv, "--points")
    argv = strip_option(argv, "-o")
    argv = strip_option(argv, "--output")
    argv = strip_option(argv, "--output-dir")
    argv = [x for x in argv if x != "--resume"]

    if args.repeat is not None:
        argv = strip_option(argv, "-r")
        argv = strip_option(argv, "--repeat")
        argv += ["-r", str(args.repeat)]
    if args.parallel_trials is not None:
        argv = strip_option(argv, "--parallel-trials")
        argv += ["--parallel-trials", str(args.parallel_trials)]

    if not args.no_steering:
        root = find_root_or_fail()
        order, keys = replayable_points(manifest, args)
        if len(keys) == 0:
            report(
                LogLevel.FATAL,
                "run {} measured nothing to replay".format(args.run),
                hint="`yuclid replay {} --no-steering` runs the command it was "
                "given instead".format(args.run),
            )
        # the points decide what runs, so anything else that would have is out
        argv = strip_list_option(argv, "-s")
        argv = strip_list_option(argv, "--select")
        argv = strip_list_option(argv, "-p")
        argv = strip_list_option(argv, "--presets")
        path = workspace.write_point_set(
            root,
            "{:%Y%m%d-%H%M%S}".format(datetime.now()),
            yuclid.run.points_of_keys(order, keys),
            replay_of=args.run,
        )
        # the id as well, so the run records where its points came from
        argv += ["--points", path, "--replay", args.run]
        report(
            LogLevel.INFO,
            "replaying {} point(s) of run {}".format(len(keys), args.run),
            path,
        )

    report(LogLevel.INFO, "replaying", " ".join(argv))
    yuclid.run.launch(as_run(parser, argv))


def replayable_points(manifest, args):
    """The points to replay: what the run measured, narrowed if asked.

    `--select` here is a filter on that list rather than on the configured
    space, since the list is what the replay is about.
    """
    import yuclid.run

    order = plan_order(manifest["directory"])
    keys = yuclid.run.measured_points(manifest["directory"])
    if not args.select:
        return order, keys

    wanted = yuclid.run.parse_coordinates(args.select, what="selector")
    unknown = [dim for dim in wanted if dim not in order]
    if unknown:
        report(
            LogLevel.FATAL,
            "unknown dimension(s) in selector: {}".format(", ".join(unknown)),
            hint="that run has {}".format(", ".join(order)),
        )
    index = {dim: i for i, dim in enumerate(order)}
    kept = [
        key
        for key in keys
        if all(key[index[dim]] in values for dim, values in wanted.items())
    ]
    if len(kept) == 0:
        report(
            LogLevel.FATAL,
            "no point of that run matches the selection",
            " ".join(args.select),
        )
    report(
        LogLevel.INFO,
        "keeping {} of {} point(s)".format(len(kept), len(keys)),
        " ".join(args.select),
    )
    return order, kept


def plan_order(directory):
    """The dimensions of a recorded run, in the order it ran them."""
    for record in workspace.read_progress(directory):
        if record["type"] == "plan":
            return list(record["order"])
    report(
        LogLevel.FATAL,
        "that run never recorded a plan, so its points are not known",
        hint="replay it with --no-steering to run the command it was given",
    )


def strip_list_option(argv, name):
    """Drop `name a b c` from a command line: an option that takes a list.

    `strip_option` removes one value, which is wrong for the options that take
    as many as they are given — everything up to the next option belongs to it.
    """
    cleaned, dropping = [], False
    for token in argv:
        if token == name:
            dropping = True
            continue
        if dropping:
            if token.startswith("-"):
                dropping = False
            else:
                continue
        if token.startswith(name + "="):
            continue
        cleaned.append(token)
    return cleaned


def as_run(parser, argv):
    """Parse a `run` command line, remembering what it was.

    The run records this rather than the `finish` or `replay` that led to it,
    so that the run it produces can itself be finished or replayed.
    """
    parsed = parser.parse_args(argv)
    parsed.origin_argv = list(argv)
    return parsed


def strip_option(argv, name):
    """Drop `name VALUE` from a command line, so replays do not accumulate."""
    cleaned, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == name:
            skip = True
            continue
        if token.startswith(name + "="):
            continue
        cleaned.append(token)
    return cleaned


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
    where = ""
    for record in reversed(workspace.read_progress(directory)):
        if record["type"] == "point.started":
            where = " " + ".".join(record["key"])
            break
    name = manifest.get("name")
    return "{}  {}  {}{}".format(
        "{} ({})".format(manifest["id"], name) if name else manifest["id"],
        state,
        format_counts(directory),
        where,
    )

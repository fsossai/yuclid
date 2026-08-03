"""The commands that look at runs and steer them.

Looking reads the run's files, which works the same whether it is still going or
finished long ago. Steering calls the run over its socket, which only a live run
answers.
"""

from yuclid.log import LogLevel, report
import yuclid.workspace as workspace
import os


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
    """How far a run got, as (completed, total), or None when it never said."""
    total, done = None, 0
    for record in workspace.read_progress(directory):
        if record["type"] == "plan":
            total = record.get("total")
        elif record["type"] == "point.finished":
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

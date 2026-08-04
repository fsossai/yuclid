"""`yuclid describe`: what is in a result dataset.

The output is a plain report on standard output rather than a stream of
diagnostics, so it can be read as a table, piped or pasted.
"""

from yuclid.log import report, LogLevel
import yuclid.log
from yuclid.plot import (
    validate_files,
    locate_files,
    generate_dataframe,
    combine_dimensions,
    explode_array_metrics,
)
import itertools
import glob
import json
import sys
import os
import numpy as np
import pandas as pd


# how many values of a dimension to name before giving up and counting
VALUE_LIMIT = 8
# how many missing points to name when no pattern explains them
POINT_LIMIT = 5


def paint(text, colour):
    style = yuclid.log._state.get("style", {})
    return "{}{}{}".format(style.get(colour, ""), text, style.get("none", ""))


def format_number(value):
    if isinstance(value, (int, np.integer)) or float(value).is_integer():
        return "{:g}".format(value)
    return "{:.6g}".format(value)


def describe_values(series):
    """Name a dimension's values, or count them when there are too many."""
    values = list(dict.fromkeys(series.tolist()))
    if len(values) > VALUE_LIMIT:
        shown = ", ".join(str(v) for v in values[:VALUE_LIMIT])
        return "{}, ... ({} more)".format(shown, len(values) - VALUE_LIMIT)
    return ", ".join(str(v) for v in values)


def describe_missing(df, dimensions):
    """Explain the combinations the dataset does not hold.

    A count on its own says little, so the absences are attributed to whole
    slices of the space wherever one accounts for them: a condition that
    carved out a region shows up as "every point with strategy=autocommit and
    batch=100". Whatever no slice explains is named point by point.
    """
    if len(dimensions) == 0:
        return 0, [], []

    domains = {d: list(dict.fromkeys(df[d].tolist())) for d in dimensions}
    expected = set(itertools.product(*[domains[d] for d in dimensions]))
    observed = set(map(tuple, df[dimensions].drop_duplicates().values))
    missing = expected - observed
    if len(missing) == 0:
        return 0, [], []

    # a slice is a pair or a single (dimension, value) constraint; the ones
    # whose every point is absent are what the reader wants to see
    index = {d: i for i, d in enumerate(dimensions)}
    candidates = []
    singles = [((d, v),) for d in dimensions for v in domains[d]]
    pairs = [
        ((a, va), (b, vb))
        for a, b in itertools.combinations(dimensions, 2)
        for va in domains[a]
        for vb in domains[b]
    ]
    for slice_ in singles + pairs:
        covered = {
            point
            for point in expected
            if all(point[index[d]] == v for d, v in slice_)
        }
        if covered and covered <= missing:
            candidates.append((slice_, covered))

    # keep the widest explanations, and drop those already covered by them
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    patterns, explained = [], set()
    for slice_, covered in candidates:
        if covered <= explained:
            continue
        patterns.append((slice_, len(covered)))
        explained |= covered

    unexplained = sorted(missing - explained)
    return len(missing), patterns, unexplained


def find_run(files):
    """The run that wrote these results, when one in this directory did.

    A dataset is a plain file and says nothing about how it was made, but the
    run directory holds a hard link to that very file, so the run can be found
    by inode. Only one run is looked for: merging several datasets makes the
    question meaningless.
    """
    import yuclid.workspace as workspace

    root = workspace.find_root()
    if root is None or len(files) != 1:
        return None
    return workspace.run_of_output(root, files[0])


def last_plan(directory):
    import yuclid.workspace as workspace

    plan = None
    for record in workspace.read_progress(directory):
        if record["type"] == "plan":
            plan = record
    return plan


def missing_from_plan(df, dimensions, plan):
    """What the run set out to measure and did not, and what became of it.

    Inferring absences from the data alone cannot see a value that is missing
    everywhere: if every point with a=2 failed, the dataset simply has no a=2
    and its cartesian product looks complete. The plan knows better, and knows
    why each point is absent.
    """
    if plan is None or sorted(plan["order"]) != sorted(dimensions):
        return None

    order = plan["order"]
    observed = {
        tuple(str(v) for v in row) for row in df[order].drop_duplicates().values
    }
    missing = [
        (tuple(str(v) for v in p["key"]), p["status"])
        for p in plan["points"]
        if tuple(str(v) for v in p["key"]) not in observed
    ]
    return order, len(plan["points"]), sorted(missing)


def name_of(manifest):
    who = manifest["id"]
    if manifest.get("name"):
        who += " ({})".format(manifest["name"])
    return who


def suggest_files():
    """Say what describe reads, and name a file if one is lying around."""
    hints = ["it reads the result files `yuclid run` writes"]
    here = sorted(
        glob.glob("*.yuclid.jsonl") + glob.glob("*.yuclid.csv"),
        key=os.path.getmtime,
        reverse=True,
    )
    if here:
        hints.append("try: yuclid describe {}".format(here[0]))
    report(LogLevel.FATAL, "no file to describe", hint=hints)


def write_points(df, dimensions, args):
    """The points this dataset holds, as a file `yuclid run --points` takes.

    A result file is a record of which points were actually measured, so it is
    the natural place to get a point list from: what came out of a run is what
    another run would have to do to produce the same dataset. Rows are grouped
    by their coordinates and written in the order they first appear.
    """
    if len(dimensions) == 0:
        report(
            LogLevel.FATAL,
            "these results have no dimensions to make points from",
            hint="every column looks like a metric; name the metrics with -y "
            "so that the rest can be read as coordinates",
        )

    seen, points = set(), []
    for row in df[dimensions].itertuples(index=False, name=None):
        key = tuple(str(value) for value in row)
        if key in seen:
            continue
        seen.add(key)
        points.append({dim: [value] for dim, value in zip(dimensions, key)})

    body = json.dumps({"points": points}, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(body)
        return
    with open(args.output, "w") as f:
        f.write(body)
    report(
        LogLevel.INFO,
        "{} point(s) written to".format(len(points)),
        args.output,
        hint="run them with `yuclid run --points {}`".format(args.output),
    )


def launch(args):
    if len(args.files) == 0:
        suggest_files()

    ctx = {"args": args}
    validate_files(ctx)
    locate_files(ctx)
    generate_dataframe(ctx)
    combine_dimensions(ctx)
    explode_array_metrics(ctx)

    df = ctx["df"]
    metrics = df.select_dtypes(include=[np.number]).columns.tolist()
    if args.y:
        unknown = [y for y in args.y if y not in df.columns]
        if unknown:
            report(
                LogLevel.FATAL,
                "unknown column: {}".format(", ".join(unknown)),
                hint="available columns: {}".format(", ".join(df.columns)),
            )
        metrics = list(args.y)
    dimensions = [c for c in df.columns if c not in metrics]

    if getattr(args, "points", False):
        write_points(df, dimensions, args)
        return

    lines = []
    def counted(n, word):
        return "{} {}{}".format(n, word, "" if n == 1 else "s")

    lines.append(
        "{}, {}, {}".format(
            counted(len(df), "record"),
            counted(len(dimensions), "dimension"),
            counted(len(metrics), "metric"),
        )
    )

    def labelled(name, count, colour, width):
        """`name (count) :` with only the name coloured, the colon aligned."""
        label = "{} ({})".format(name, count)
        return "  {}{} : ".format(
            paint(name, colour), label[len(name):].ljust(width - len(name))
        )

    if dimensions:
        width = max(len("{} ({})".format(d, df[d].nunique())) for d in dimensions)
        lines.append("")
        lines.append("dimensions")
        for d in dimensions:
            lines.append(
                labelled(d, df[d].nunique(), "bold", width)
                + describe_values(df[d])
            )

    if metrics:
        samples = {m: pd.to_numeric(df[m], errors="coerce").dropna() for m in metrics}
        width = max(len("{} ({})".format(m, len(samples[m]))) for m in metrics)
        lines.append("")
        lines.append("metrics")
        for m in metrics:
            values = samples[m]
            head = labelled(m, len(values), "green", width)
            if len(values) == 0:
                lines.append(head + "no numeric samples")
                continue
            lines.append(
                "{}{} … {}  (median {})".format(
                    head,
                    format_number(values.min()),
                    format_number(values.max()),
                    format_number(values.median()),
                )
            )

    manifest = find_run(ctx["local_files"])
    planned = None
    if manifest is not None:
        planned = missing_from_plan(df, dimensions, last_plan(manifest["directory"]))

    if planned is not None:
        order, combinations, missing = planned
        lines.append("")
        lines.append("from run {}".format(name_of(manifest)))
        if len(missing) == 0:
            lines.append("  every point it planned was recorded")
        else:
            lines.append(
                "  missing: {} of {} points ({:.0%})".format(
                    len(missing), combinations, len(missing) / combinations
                )
            )
            shown = missing if args.missing else missing[:POINT_LIMIT]
            for point, status in shown:
                lines.append(
                    "    {}  {}".format(
                        ", ".join(
                            "{}={}".format(paint(d, "bold"), v)
                            for d, v in zip(order, point)
                        ),
                        status,
                    )
                )
            if not args.missing and len(missing) > len(shown):
                lines.append(
                    "    ... and {} more, with --missing".format(
                        len(missing) - len(shown)
                    )
                )
            lines.append(
                "  `yuclid finish {}` runs what it did not record".format(
                    manifest["id"]
                )
            )
        print("\n".join(lines))
        return

    total, patterns, unexplained = describe_missing(df, dimensions)
    if total > 0:
        combinations = int(np.prod([df[d].nunique() for d in dimensions]))
        lines.append("")
        lines.append(
            "missing: {} of {} combinations ({:.0%})".format(
                total, combinations, total / combinations
            )
        )
        for slice_, count in patterns:
            lines.append(
                "  every point with {}  ({})".format(
                    " and ".join(
                        "{}={}".format(paint(d, "bold"), v) for d, v in slice_
                    ),
                    count,
                )
            )
        if unexplained:
            shown = unexplained if args.missing else unexplained[:POINT_LIMIT]
            for point in shown:
                lines.append(
                    "  {}".format(
                        ", ".join(
                            "{}={}".format(paint(d, "bold"), v)
                            for d, v in zip(dimensions, point)
                        )
                    )
                )
            if not args.missing and len(unexplained) > len(shown):
                lines.append(
                    "  ... and {} more, with --missing".format(
                        len(unexplained) - len(shown)
                    )
                )

    print("\n".join(lines))

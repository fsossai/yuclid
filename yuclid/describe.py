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

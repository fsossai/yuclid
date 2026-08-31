---
name: "yuclid-plot"
description: "Suggest ways to visualize a yuclid result file, defaulting to the workspace's most recent run when none is named"
---

# Suggesting views of a yuclid dataset

A yuclid dataset is a table whose columns split in two: **dimensions** (what was
varied) and **metrics** (what was measured). A view is always the same shape:

- **X** — one dimension, along the horizontal axis. Mandatory.
- **Z** — one dimension, the series: one bar colour or one line per value.
- **Y** — one metric.
- Everything else is a **free dimension**: the viewer shows one slice at a time
  and the arrow keys walk through the rest.

So proposing a view means answering: *which two dimensions belong on the plot,
and which metric answers the question?* Everything else follows.

## 1. Establish which file

Everything below is about one dataset, so start from the file. If the user
named one, use it and skip the rest of this section.

Invoked with no arguments, do not ask which file and do not `ls` for one:
**default to the most recent run of the workspace**. The workspace is the
directory holding `yuclid.json`, which is the working directory unless the user
pointed somewhere else, and yuclid will name that run's results file itself:

```sh
yuclid runs --last          # add --workspace DIR for a workspace elsewhere
```

That prints one absolute path and nothing else, which is what to pass on to
`describe` and to every command suggested below.

**Check that the path exists before using it.** It is the destination the run
recorded, not proof of a file: a run started from the web UI never writes one,
and one written months ago may have been moved or cleaned away. Every run also
keeps its own copy, which is always there and always JSON Lines whatever
format the destination was:

```sh
yuclid runs                 # run ids, state, progress, destination — newest first
yuclid describe .yuclid/runs/RUN_ID/results.jsonl
```

So the order is: `yuclid runs --last`; if that path is missing, take the top id
from `yuclid runs` and use `.yuclid/runs/<id>/results.jsonl` instead — under the
workspace, so `DIR/.yuclid/runs/<id>/results.jsonl` when `--workspace DIR` is in
play. Both are ordinary result files; `describe`, `plot`, `tplot` and `stats`
all read either one.

Say which run you settled on — its id, and whether it finished — before
suggesting anything. The newest run is a default, not a certainty: if
`yuclid runs` shows it stopped early or covering a different space from the
one the user is asking about, say so and offer the run that fits instead.

Only where there is no `.yuclid` at all — not a workspace, or a bare directory
of result files someone copied there — fall back to looking for the files
themselves:

```sh
ls -1 *.yuclid.jsonl *.yuclid.csv
```

If exactly one turns up, say which one you are using and carry on. If several
do, or none, **ask the user which file to look at** rather than guessing — the
suggestions are worthless if they describe another experiment.

## 2. Read the schema

Never guess column names. Ask yuclid:

```sh
yuclid describe FILE
```

It reports the dimensions with their values, the metrics with their range, and
which combinations of the space the dataset does not hold — a run cut short, or
a condition that carved a region out.

## 3. Work out what the experiment was about

Do not pick axes mechanically. The names of the dimensions and the values they
take say what was being investigated, and therefore what the person who ran it
expects to see. Read them and infer:

- Values that are numbers growing in a sequence — `1, 2, 4, 8`, `512, 1024,
  2048` — are a **scale someone was pushing**. The question is how the metric
  behaves along it, so this belongs on X, usually with `-l`.
- Values that are names of things — `gcc, clang`, `dot, rows, columns`,
  `WAL, DELETE, OFF` — are **alternatives being compared**. The question is
  which one wins, so this belongs on Z, side by side.
- A value that reads like an off state — `none`, `0`, `baseline`, `default` —
  is the **reference** the others are meant to be measured against, which is
  what the normalization flags are for.
- Values that look interchangeable — `seed`, `run`, `rep`, machine numbers —
  are **noise, not a subject**. Leave such a dimension free rather than putting
  it on an axis.
- Metric names say which direction is good: `seconds`, `latency`, `peak_rss`
  are better when small; `throughput`, `rows_per_second`, `speedup` when large.

From that, ask what the obvious question is. A configuration comparing three
implementations over four input sizes was almost certainly run to find out
which implementation is fastest and whether that holds as the input grows —
so propose exactly that, then the variations.

When the meaning genuinely does not settle it, fall back on shape: the
dimension with the most values on X, the one with fewest on Z, since a legend
of two or three reads well and one of ten does not. X, Z and Y must be three
different columns, and Y must be numeric.

A dimension that is not part of the question is better filtered away with
`-f dim=value` than left free.

## 4. The flags worth suggesting

| Flag | What it does |
|---|---|
| `-f dim=v1,v2` | keep only these values of a dimension |
| `-L dim=value` | pin a free dimension instead of cycling it |
| `-C a,b` | merge two dimensions into one axis named `a_b` |
| `-l` | lines instead of bars — for an X with a numeric order |
| `-A` | write the value on each bar; `--annotate-max` / `--annotate-min` for just the extremes |
| `-g` | add a geomean column summarizing all X values (bars only) |
| `-m` | spread around each point: `pi,95`, `sd,2`, `iqr`, `range`, `mad`, `none` |
| `-u` | unit for the Y label |
| `--colorblind` | colour-vision-safe palette |

## 5. Normalization: showing a comparison rather than absolutes

Three mutually exclusive forms. Getting the coordinate right is the part that
trips people up:

- `-R x=value z=value` — divide everything by that one point. Needs **both**
  the X and the Z coordinate.
- `-X z=value` — within each X category, divide by that Z series. Names **Z**
  only.
- `-Z x=value` — within each Z series, divide by that X category. Names **X**
  only.

The reference is normally the value step 3 identified as the off state or the
baseline. Add `-r` to invert the ratio, turning "0.5× the time" into "2×
faster" — which is what a reader expects when the metric is a time. A speedup
against a baseline therefore reads:

```sh
yuclid plot FILE -x variant -z tile -R variant=dot tile=none -r -A
```

## 6. Repetitions

If the run used `-r N`, each point carries several samples. The viewers reduce
them to a median with a spread band; `-m` chooses the band. To look at the
distribution itself rather than a summary, `yuclid stats` draws one histogram
per group. It needs `-y`; `-z` defaults to the narrowest dimension.

```sh
yuclid stats FILE -y seconds -z pattern
```

## 7. What to hand back

Three to five commands, most useful first, each with one line saying what
question it answers — phrased in the terms of the experiment, not of the flags:
"how the three kernels compare as the matrix grows", not "seconds by size
grouped by variant".

**Every command is `yuclid plot`.** `tplot` takes the same arguments and draws
in the terminal instead of a window, but suggest it only where the user has
asked for it by name, or asked to stay in the terminal, or said they are over
SSH or without a display. Absent that, do not offer it as an alternative and do
not mention it — one clear command per question is the point.

Keep every name exactly as the schema spells it, and do not invent flags. If
`describe` reported a large part of the space missing, say so: a view whose X
or Z is mostly absent will look emptier than the user expects.

Two caveats worth passing on:

- With a **CSV** file, a dimension whose values are numbers is indistinguishable
  from a metric, so name the metrics with `-y` explicitly.
- `-g` and `-l` cannot be used together, and `-g` says nothing when Z has a
  single value.

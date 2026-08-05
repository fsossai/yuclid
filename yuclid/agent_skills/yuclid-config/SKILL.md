---
name: "yuclid-config"
description: "Author or fix a yuclid.json configuration"
---

# Writing a `yuclid.json`

Yuclid takes the Cartesian product of the dimensions declared in `space`, runs the
`trials` commands once per point, and scrapes numbers out of each trial's output with
the `metrics` commands. The result is one JSON Lines record per point.

Work in this order: **space → trials → metrics → setup/env → order/presets**, then
verify with `yuclid run --dry-run`.

## Top-level fields

Exactly these seven are recognized. Anything else emits `unknown field in configuration`
and is dropped.

| Field | Type | Purpose |
|---|---|---|
| `env` | array or object | Shell variables exported to every command, in groups |
| `setup` | object | `global` and/or `point` commands run before the trials |
| `space` | object | Dimension → list of points |
| `trials` | string or array | The experiment command(s) |
| `metrics` | array or object | How to extract numbers from a trial |
| `presets` | object | Named subspaces selectable with `-p` |
| `order` | array | Which dimensions vary slowest |

`trials` is effectively required (empty ⇒ fatal). Missing `metrics` only warns, but then
nothing is collected.

## Substitution

Two scopes with **different, non-interchangeable** variable sets.

**Point scope** — `trials`, `metrics`, `setup.point`, and a point's own `setup`:

- `${yuclid.dim}` — the point's `value` (alias for `${yuclid.dim.value}`)
- `${yuclid.dim.name}` — the point's `name`
- `${yuclid.@}` — unique identifier for the current trial. Yuclid always creates
  `${yuclid.@}.out` and `${yuclid.@}.err` (under `--temp-dir`, default `.yuclid`), so
  metric commands read those files.

**`${yuclid.workspace}` is reserved and available in every scope** — trials, metrics,
setup, `env`, and a compiled script. It is the directory the run records itself in:
`./.yuclid` normally, and whatever `--workspace DIR` says otherwise. Use it for anything
the configuration generates and means to keep beside the runs — built binaries,
generated corpora, a shared scratch directory:

```json
{ "setup": { "global": ["mkdir -p ${yuclid.workspace}/data"] } }
```

Because it follows `--workspace`, the same configuration puts its data on a scratch
filesystem when the state is sent there, without a second flag or an environment
variable to keep in step. A dimension called `workspace` is refused, since one of the two
would shadow the other.

**Global scope** — `env` and `setup.global`, which run once and have no current point:

- `${yuclid.dim.values}` — all values of the dimension, space-joined
- `${yuclid.dim.names}` — all names, space-joined

Using a point variable in global scope is a fatal error, and vice versa. Dimension names
must match `[a-zA-Z0-9_]+`. Plain `$VAR` / `${VAR}` shell expansion still works
everywhere — including references to keys defined in `env`.

## `space`

Keys are dimension names, values are lists of points. Four accepted forms:

```json
{
  "space": {
    "cpuid": [0, 1, 2, 3],

    "size": [
      { "name": "small",  "value": "10M" },
      { "name": "medium", "value": "20M" }
    ],

    "threads:py": "list(range(1, 5))",

    "nthreads": null
  }
}
```

- **Scalars** (string/int/float) become `{name: str(value), value: value}`.
- **Objects** require `value`; optional `name` (defaults to `str(value)`), `condition`,
  and `setup`. Any other key warns. Valid keys: `name`, `value`, `condition`, `setup`.
  - `condition` is a Python expression evaluated per point, written over *other*
    dimensions as `yuclid.<dim>`, e.g. `"yuclid.threads > 1"`. It sees the point's
    **value**, with the JSON type preserved — an int in `space` compares as an int.
    Points whose condition is false are carved out of the space. Default `"True"`.
  - `setup` is a command or list of commands run once during global setup if that point
    is in the subspace.
  - **Names may be duplicated** — that is the idiom for picking a different value per
    region of the space:
    ```json
    "dataset": [
      { "name": "in", "value": "${data_dir}/a.dat", "condition": "int(yuclid.nthreads) == 1" },
      { "name": "in", "value": "${data_dir}/b.dat", "condition": "int(yuclid.nthreads) > 1" }
    ]
    ```
    (`int(...)` because `nthreads` here is a `null` dimension filled from the command
    line — see the gotcha below.)
- **`"dim:py"`** — the value is a Python expression `eval`'d at load time and must
  produce a list; the dimension is registered as `dim` (suffix stripped). Only builtins
  and `run.py`'s imports are in scope.
- **`null`** — declares the dimension but leaves it undefined, forcing the user to supply
  it on the command line: `yuclid run -s nthreads=1,7,14`. See below.

### A dimension the configuration deliberately does not fix

```json
{ "space": { "size": [512, 1024], "nthreads": null } }
```

```sh
yuclid run -s nthreads=1,7,14
```

Reach for `null` when the values are not a property of the experiment but of the
occasion it is run on: how many threads this machine has, which GPU is free, where a
dataset happens to live, which compilers are installed. Those belong to the person
running it, not to the question being asked.

Writing `[1, 7, 14]` instead would bake one machine into the configuration, and the next
person would edit the file to run it — the file everyone shares, and the one the results
are supposed to be comparable across. `null` says "this experiment sweeps thread counts;
which ones is up to you", and every run then records the answer it was given, so the
results say which machine they came from.

It is filled by `-s`, or by a preset that lists the values, and the run refuses to start
without them. Three things follow:

- values arrive as **strings**, so a condition comparing them numerically needs a cast —
  `"int(yuclid.nthreads) > 1"`, not `"yuclid.nthreads > 1"`;
- a preset may supply them, but a `*` glob cannot: there is nothing declared to match, so
  list the literal values;
- `yuclid serve` offers the dimension as a free-text field rather than chips, for the
  same reason — it has nothing to put on the chips.

## `trials`

A single string, or a list whose items are strings or objects:

```json
{
  "trials": [
    "time -p ${yuclid.compiler}/prog.out ${yuclid.dataset} > ${yuclid.@}.out",
    {
      "command": ["perf stat -e cache-misses", "./prog.out ${yuclid.dataset}"],
      "condition": "yuclid.compiler == 'g++'",
      "metrics": ["cache_misses"]
    }
  ]
}
```

- `command` is required in object form; a list of strings is joined with a single space.
- `condition` — Python expression over `yuclid.<dim>`, default `"True"`.
- `metrics` — names of the metrics this trial enables. Omit (or `null`) to enable all.
  Referencing an undeclared metric name is fatal. A metric no trial enables is never
  measured, whatever its condition says — if every trial names its metrics, make sure one
  of them names yours.
- Unknown keys warn. Valid keys: `command`, `condition`, `metrics`.

Trials do not need to redirect output: stdout and stderr are already captured into
`${yuclid.@}.out` and `${yuclid.@}.err`.

**A metric reads the output of the trial that declared it.** Each trial gets its own
`${yuclid.@}` (suffixed `_trial0`, `_trial1`, …), and a metric is evaluated against the
capture of the trial whose `metrics` list names it. So several trials may run for one
point, as long as their `metrics` lists are **disjoint** — that is how one program gets
measured two ways in a single run.

A metric that reads `${yuclid.@}` and is enabled by more than one trial is a **fatal
configuration error**, reported before anything runs:

```
ERROR: these metrics are enabled by more than one trial: m (2 trials at 1)
HINT: check the conditions of the ambiguous metrics or trials
```

Beware that a trial with **no** `metrics` list declares them all, so adding one beside a
trial that names its metrics makes every one of them ambiguous.

## `metrics`

Each metric command must print one or more numbers separated by whitespace or newlines.
Multiple numbers are averaged, or kept as an array with `--fold`.

Full form:

```json
{
  "metrics": [
    {
      "name": "time.real",
      "command": "grep real ${yuclid.@}.err | grep -oE '[0-9]+\\.[0-9]+'",
      "condition": "yuclid.size != 'large'"
    }
  ]
}
```

Shorthand map form (no conditions):

```json
{
  "metrics": {
    "time.real": "grep real ${yuclid.@}.err | grep -oE '[0-9]+\\.[0-9]+'"
  }
}
```

`name` and `command` are required; `command` may be a list of strings (space-joined).
Unknown keys warn. Valid keys: `name`, `command`, `condition`, `default`. Metric names
become the column names consumed by `yuclid plot -y`.

**`default`** is the number to record where every declaration of that name is conditioned
away, so that a point a condition carved out of one metric still has a row as complete as
the others:

```json
{ "name": "visits_per_op",
  "command": "awk ...",
  "condition": "yuclid.operation in ['lookup', 'absent']",
  "default": 0 }
```

It must be a number — it lands in the column the command's output would have — and it is
used only when *no* declaration of that name applies. Nothing is executed for it: a point
whose metrics are all defaults runs no trial at all and still gets a row. A default on a
metric that is unconditional somewhere can never fire, and is reported when the
configuration is read.

## `env`

A **list of groups**, resolved in order. A group is resolved as a whole, so nothing in it
can see anything else in it; a group may refer to the groups before it and to any
inherited variable.

```json
{ "env": [
    { "ALL_SIZES": "${yuclid.size.values}" },
    { "data_dir": "${yuclid.workspace}/data" }
] }
```

That is what fixes the order: an object has none to rely on, and a formatter is free to
rearrange one. A plain object is accepted as a list of one group, which is what most
configurations are — a handful of constants that refer to nothing:

```json
{ "env": { "CFLAGS": "-O3 -std=c11" } }
```

An object with more than one entry warns, because nothing in it can depend on anything
else in it and there is no way to say otherwise.

Each value is expanded by the shell (`echo "<value>"`). Only global
`${yuclid.*.values}` / `${yuclid.*.names}` are allowed here.

## `setup`

**Setup exists to put in place everything the points will need.** A trial names files —
a binary, a generated input, a directory to write into — and a run is only reproducible
if every one of them is something setup makes rather than something that happened to be
lying around on the machine where the configuration was written.

So read the configuration backwards once it is written: take every path in `trials` and
`metrics`, resolve the `${yuclid.*}` in it, and find the setup command that creates it.
A path with no such command is the line that will fail on somebody else's machine, or on
yours after a `git clean`.

```json
{
  "setup": {
    "global": ["ulimit -s 1048576"],
    "point": [
      { "on": ["compiler"], "command": "mkdir -p ${yuclid.compiler}" },
      {
        "on": ["compiler"],
        "command": "make prog.out CXX=${yuclid.compiler} OUTDIR=${yuclid.workspace}/build/${yuclid.compiler}",
        "parallel": true
      }
    ]
  }
}
```

- `global` — a string or list of commands, run once before everything. Global scope.
- `point` — items run once per combination of the dimensions listed in `on`, not once per
  full point. Valid keys: `command`, `on`, `parallel` (others warn); `command` required.
  - `on` omitted ⇒ the entire space. Entries default to `dim.values`; write `dim.names`
    to iterate names instead.
  - `parallel` — `false` (default), `true` (parallelize over every dimension in `on`), or
    an explicit list of dimensions.
- Global setup runs before point setup. `--no-setup` skips both.

Prefer `point` with a narrow `on` over `global` for anything per-point: a binary built
once per compiler belongs to `on: ["compiler"]`, not to a global step that rebuilds it,
and not to a full-space step that builds it once per thread count as well.

## `order`

```json
{ "order": ["compiler", "dataset", "nthreads"] }
```

Listed dimensions are moved to the end of the iteration order, so the **last** entry
varies fastest — above, all `nthreads` are swept for a given dataset, then datasets, then
compilers. Names must exist in `space` (fatal otherwise). `--order` overrides at runtime.

## `presets`

Named subspaces, selected with `yuclid run -p quick`. Each preset maps a dimension to a
list of point **names** (not values); `*` acts as a glob over names:

```json
{
  "presets": {
    "quick": { "size": ["small"], "compiler": ["clang*"] },
    "sweep": { "nthreads": [1, 2, 4] }
  }
}
```

Globs cannot be used on `null` dimensions — there list the literal values instead.
An unknown dimension is fatal; an unknown name is an error.

## Patterns for sophisticated configurations

These are the idioms that make a config scale to a real benchmark suite (dozens of
programs, several toolchains, per-program inputs and metric formats) instead of collapsing
into one config per program.

### Trials as a dispatch table

When different regions of the space need a different launch wrapper, write one trial per
region with **mutually exclusive conditions** and let each declare the metrics it feeds:

```json
{
  "trials": [
    { "command": ["OMP_NUM_THREADS=${yuclid.nthreads}", "/usr/bin/time -v",
                  "${yuclid.workspace}/bin/${yuclid.program}_${yuclid.impl}.out", "${yuclid.input}", "2>&1"],
      "condition": "yuclid.program in ['prog1', 'prog2']",
      "metrics": ["time", "space"] },
    { "command": ["OMP_NUM_THREADS=${yuclid.nthreads}", "/usr/bin/time -v",
                  "${yuclid.workspace}/bin/${yuclid.program}_${yuclid.impl}.out", "${yuclid.input}", "2>&1"],
      "condition": "yuclid.program in ['prog3', 'prog4']",
      "metrics": ["time", "space"] },
    { "command": ["LD_PRELOAD=${yuclid.workspace}/nmallocs/libmymalloc.so",
                  "${yuclid.workspace}/bin/${yuclid.program}_${yuclid.impl}.out", "${yuclid.input}"],
      "metrics": ["nmallocs"] }
  ]
}
```

A trial runs for a point only if its `condition` holds **and** at least one of its declared
metrics is currently selected. So the third trial above — an instrumented re-run under a
different `LD_PRELOAD`, with no condition — stays dormant during a normal
`yuclid run -m time space`, and `yuclid run -m nmallocs` runs it *instead of* the others.

This is what keeps the one-trial-per-point rule satisfied: partition by `condition` within
a metric group, and by `metrics` across groups. Two mutually exclusive conditions plus
disjoint metric sets means exactly one trial fires per invocation. A trial with neither a
condition nor a `metrics` list always runs and will shadow the others' output files.

Writing the command as a **list of strings** keeps a long `VAR=… wrapper … binary … args`
line reviewable; it is joined with single spaces.

### Polymorphic metrics: same name, disjoint conditions

Different programs print their timing in different formats, but you still want a single
`time` column to plot. Declare the metric several times under the same name with
conditions that partition the space:

```json
{
  "metrics": [
    { "name": "time", "command": "grep Trial ${yuclid.@}.out | grep -oE '[0-9]+\\.[0-9]+'",
      "condition": "yuclid.program in ['prog1', 'prog2']" },
    { "name": "time", "command": "grep ROI ${yuclid.@}.out | awk '{print $3}'",
      "condition": "yuclid.program in ['prog3', 'prog4']" }
  ]
}
```

Keep the conditions disjoint — with two matching at once, both run and the later one wins
the column. `-m time` selects all variants at once, since selection is by name.

### Irregular spaces: same name, different value per region

The counterpart in `space`. `input=medium` should mean a different file for every program,
so declare one point per (name, region) pair and condition it on the other dimension:

```json
{
  "input": [
    { "name": "medium", "value": "-f $graphs_dir/kron24.sg -n $nruns", "condition": "yuclid.program in ['bc']" },
    { "name": "large",  "value": "-f $graphs_dir/twitter.sg -n $nruns", "condition": "yuclid.program in ['bc']" },
    { "name": "medium", "value": "-f $graphs_dir/twitter.sg -n $nruns", "condition": "yuclid.program in ['bfs']" },
    { "name": "large",  "value": "-f $graphs_dir/urand.sg -n $nruns",   "condition": "yuclid.program in ['bfs']" }
  ]
}
```

`-s input=medium` then selects the right input for each program, and the results share one
comparable `input` column. Extra names that only make sense in one region (`large256`,
`large1024`, …) can live in the same list — they simply produce no points elsewhere.

This is the central trick for an irregular space: the Cartesian product stays rectangular,
and conditions carve out the invalid cells. Watch the `subspace size` and the dry-run point
count to confirm the carving worked.

### Tiers of variables

Long-lived configs use all of these, and they resolve at different times:

| Form | Resolved | Use for |
|---|---|---|
| `${yuclid.dim}` | by yuclid, per point | anything that varies over the space |
| `${yuclid.workspace}` | by yuclid, always | where this run keeps what it builds |
| `$var` from `env` | by the shell, once at startup | derived paths (`"graphs_dir": "${yuclid.workspace}/graphs"`) |
| `$var` inherited | by the shell, from the caller's environment | machine/run identity — `$machine`, `$nruns` |

Inherited variables never appear in the config's `env`; a driver script exports them and
then invokes `yuclid run`, which is how one config serves several machines:

```bash
export machine="local"
export nruns="3"
yuclid run -i yuclid.json --select nthreads=28 impl=base cxx=clang++ "$@"
```

Keep that driver in the repo next to the config — it is the honest record of how a run was
parameterized, and `--select`/`--metrics` lines can be commented in and out per experiment.

### Deferring expansion to an inner shell

`\\$VAR` in JSON reaches the shell as `\$VAR`, which survives yuclid's shell as the literal
text `$VAR`. Use it when a wrapper re-evaluates the command and the variable is only set at that inner level:

```json
{ "name": "large", "value": "a b c \\$OMP_NUM_THREADS" }
```

The trial sets `OMP_NUM_THREADS=${yuclid.nthreads}` as a command prefix, so an unescaped
`$OMP_NUM_THREADS` would expand *before* the assignment takes effect and pass an empty
argument. Escaping defers it until the wrapper's `eval`, when the assignment is live.

### Staged, selectively parallel point setup

Build steps usually need two stages with different fan-out, and `parallel` must exclude any
dimension whose jobs share a mutable target:

```json
{
  "setup": {
    "point": [
      { "on": ["cxx"], "command": "mkdir -p ${yuclid.workspace}/bin/$machine/${yuclid.cxx}", "parallel": true },
      { "on": ["cxx", "program", "impl"],
        "command": "make ${yuclid.program}_${yuclid.impl} OUTDIR=${yuclid.workspace}/bin/$machine/${yuclid.cxx} CXX=${yuclid.cxx}",
        "parallel": ["cxx", "program"] }
    ]
  }
}
```

`on` lists three dimensions, so the build runs once per (cxx, program, impl) — not once per
full point, which would rebuild for every thread count and malloc. `parallel` names only
`cxx` and `program`: the `impl` variants of one program write related targets and must be
serialized. Put the machine identity in the output path (`$machine`) so one checkout can
hold results from several hosts.

### Splitting a config across files

`-i` merges: `space` and `presets` merge as dicts; `env`, `trials`, `metrics` and `order`
concatenate. Two uses:

- **Metric overlays** — an extra file containing nothing but a `metrics` array of
  fine-grained timers, added only when you need them:
  `yuclid run -i yuclid.json -i yuclid.more.json -m sc.pgain sc.Kernel`
- **Variants** — a near-copy that adds a dimension (say a build-parameter `k` threaded
  through the output path and the `make` line) kept as its own file rather than conditioned
  into the main one.

Since `space` merges per key, a second file can also redefine one dimension wholesale —
e.g. narrowing `program` to a single benchmark — while inheriting everything else.

### Ordering for experimental hygiene

`order` is not only cosmetic. Put the dimension you are comparing **last** so it varies
fastest and its variants run back to back, minimizing drift in machine state (thermals,
page cache, other tenants) between the numbers you will divide:

```json
{ "order": ["cxx", "malloc", "input", "nthreads", "program", "impl"] }
```

Here two `impl` variants of the same program, input, and thread count are measured
consecutively; the expensive-to-change dimensions (`cxx`, `malloc`) vary slowest.

### Getting output into the right file

Yuclid captures stdout into `${yuclid.@}.out` and stderr into `${yuclid.@}.err`. Tools like
`/usr/bin/time -v` and `perf stat` write to stderr, so a trial ending in `2>&1` folds
everything into `.out` and every metric can grep one file. Decide once per trial, then be
consistent — a metric grepping `.err` for a trial that redirected to `.out` silently yields
nothing, and an empty metric drops the entire point from the results.

## Gotchas

- **No comments.** The config is parsed with strict `json.load`. The `//` annotations in
  the README are documentation only — a real `yuclid.json` containing them fails to parse.
  Trailing commas are also rejected.
- **Escape backslashes.** A regex like `[0-9]+\.[0-9]+` must be written `[0-9]+\\.[0-9]+`
  in JSON.
- **Quote nesting.** Metric commands run through a shell; prefer single quotes inside the
  JSON double-quoted string.
- **Multiple inputs merge.** `yuclid run -i base.json local.json` merges `space` and
  `presets` as dicts (later wins per key) and concatenates `env`, `trials`, `metrics` and
  `order` — useful for machine-specific overlays. `env` concatenating means a second file
  adds a group after the first file's, which is exactly the order you want.
- Conditions are Python, not shell: `and`/`or`/`not`, `==`, `'quoted strings'`.
- **`--select` yields strings.** Values supplied on the command line for a `null`
  dimension arrive as strings, so `"yuclid.nthreads > 1"` raises
  `TypeError: '>' not supported between instances of 'str' and 'int'`. Write
  `"int(yuclid.nthreads) > 1"`. Dimensions declared with numeric literals in `space` keep
  their type and need no cast.
- A `null` dimension must be filled by **either** `-s` or a preset that lists it — a
  preset that leaves it undefined crashes in `print_subspace` rather than reporting
  cleanly.
- **An empty metric drops the whole point.** `metric X generated an empty string` is an
  error, not a warning, and no record is written for that point. It almost always means
  the grep targeted the wrong file (`.out` vs `.err`). Give the metric a `default` if
  having no number there is a legitimate outcome.
- **`${yuclid.@}` in a metric is the id of the trial that declared it.** Trials with
  disjoint `metrics` lists therefore coexist; one that declares none of them declares
  all, and makes the rest ambiguous.
- Trial `condition`s see values, so `yuclid.program in ['bc', 'bfs']` works on a plain
  string dimension but a list membership test against a `null` dimension compares strings.
- `${yuclid.dim}` yields the *value*; use `${yuclid.dim.name}` for the label. Output
  records carry names, not values.

## Repeating until something is true

`-r N` asks for a number of repetitions, which is rarely the question. Ten runs of a
two-second point is a twenty-second answer nobody needed; ten runs of a
forty-millisecond point is noise measured ten times. `--until` lets the measurement
decide:

```sh
yuclid run --until 3s                    # give every point three seconds
yuclid run --until 'time±5%'             # until the median of `time` is that well known
yuclid run --until 3s --until 'time±5%'  # whichever is satisfied first
```

A rule is either a **duration** — `3s`, `500ms`, `2m`, `1.5s`, or a bare number of
seconds — or a **precision**, `metric±x%`, with `+-` accepted for `±` so it can be
typed. Several may be given, and the first one satisfied ends the point.

| flag | meaning |
|---|---|
| `--until RULE` | repeatable; stop when any rule holds |
| `--min-runs N` | never stop before N repetitions (default 3) |
| `--max-runs N` | never go past N (default 100) |

`-r` and `--until` are refused together: bound a rule with `--min-runs`/`--max-runs`.

The precision rule is a percentile bootstrap on the **median** — the statistic one
slow repetition does not move — over every value that metric has produced for the
point. It watches a metric, so the metric must exist and must not be excluded by
`-m`; both are fatal, and said before the run starts.

Two things follow from a count nobody knows in advance:

- the progress total counts towards `--max-runs` and **falls as points settle**, so
  `[12/400]` becomes `[12/38]`. Nothing is wrong: the total is a ceiling until the
  measurements say otherwise.
- with `--resume`, the rule judges only what *this* invocation measured, since the
  earlier repetitions left their samples nowhere. A resumed point is given a fresh
  budget, and its records add to the ones already in the file.

## Running something that is not a product

`yuclid run --points FILE` covers exactly the points a file names, instead of the product
of `space`. The configuration is still read in full — trials, metrics, setup, env,
conditions, and the declared values — but `space` stops deciding what runs. It is a mode,
so it is refused alongside `-s/--select` and `-p/--presets`.

```json
[
  { "size": ["*"], "impl": ["dot"] },
  { "size": ["1024"], "impl": ["rows"], "threads": ["1", "4"] }
]
```

Each entry is a small sub-product; the run is their union, deduplicated, in file order.
`"*"`, or a dimension left out, means every value the configuration declares for it. A
value the configuration does not have is an error naming it, and a named point a condition
excludes is reported and dropped. Every such run keeps its resolved list as `points.json`
in its run directory.

`yuclid describe RESULTS --points` writes such a file from a dataset, so
`yuclid run --points <that>` re-runs exactly what a result file contains. `yuclid replay`
is built on the same mechanism.

## Keeping state elsewhere

`--workspace DIR` names the directory Yuclid records itself in, instead of `./.yuclid`:
the runs, their logs and their captured output. `yuclid run`, `yuclid serve` and every
command that reads or steers a recorded run take it. For `yuclid serve` the workspace is
the whole of what it is about — the configuration it offers and the directory it starts
runs in come from there too — so a workspace can be moved or served on its own.

## Verify before handing off

Always dry-run. It parses and normalizes the config, prints the `env` bindings, the
resolved subspace per dimension, its total size, and every point it would visit in order —
without executing anything:

```
yuclid run --dry-run
yuclid run --dry-run -s size=small cpuid=0     # shrink the space first
yuclid run -s size=small                       # then a real smoke run
```

The dry run prints every setup command and every point, which is what makes the last
check possible: **every file the trials and metrics name should be one that setup
creates.** Walk the printed commands and tick off the paths in the printed trials; a path
nothing accounts for is the one that will fail somewhere else. Running the smoke test in
a fresh clone, or after deleting whatever setup builds, turns that check into a proof.

Then check the JSONL and plot it:

```
yuclid tplot results.jsonl -x size -z compression -y time.real -A
```

Other `yuclid run` flags worth mentioning: `-o/--output`, `--output-dir`, `-p` (presets),
`-m` (subset of metrics — which also decides which trials fire), `-r N` (repeat each
point), `--parallel-trials [N]`, `--fold`, `--no-setup`, `--temp-dir`, `--points FILE`,
`--workspace DIR`, `--abort-on-error`, `--resume`, `--until RULE` with
`--min-runs`/`--max-runs`.

## Reference example

Complete, runnable, parse-clean:

```json
{
  "space": {
    "size": [
      { "name": "small",  "value": "10M" },
      { "name": "medium", "value": "20M" },
      { "name": "large",  "value": "50M" }
    ],
    "cpuid": [0, 1, 2, 3],
    "compression": [
      { "name": "lowest",  "value": 1 },
      { "name": "highest", "value": 9 }
    ]
  },
  "trials": [
    "time -p taskset -c ${yuclid.cpuid} head -c ${yuclid.size} /dev/urandom | gzip -${yuclid.compression} >/dev/null"
  ],
  "metrics": [
    { "name": "time.real", "command": "cat ${yuclid.@}.err | grep real | grep -oE '[0-9]+\\.[0-9]+'" },
    { "name": "time.sys",  "command": "cat ${yuclid.@}.err | grep sys | grep -oE '[0-9]+\\.[0-9]+'" }
  ],
  "order": ["size", "compression"]
}
```

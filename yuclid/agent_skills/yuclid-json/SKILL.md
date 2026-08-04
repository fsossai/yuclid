---
name: yuclid-json
description: Author or fix a yuclid.json configuration for `yuclid run` — the experiment space, trials, and metric extraction. Use whenever the user mentions yuclid.json, a yuclid config, `yuclid run`, or asks to set up a parameter sweep / combinatorial experiment / benchmark grid whose results feed `yuclid plot`, `yuclid tplot`, or `yuclid stats`.
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
| `env` | object | Shell variables exported to every command |
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
  it on the command line: `yuclid run -s nthreads=1,7,14`.

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
  Referencing an undeclared metric name is fatal.
- Unknown keys warn. Valid keys: `command`, `condition`, `metrics`.

Trials do not need to redirect output: stdout and stderr are already captured into
`${yuclid.@}.out` and `${yuclid.@}.err`.

**Aim for exactly one trial per point.** Every compatible trial is executed, but each gets
its own `${yuclid.@}` (suffixed `_trial0`, `_trial1`, …) and the metrics are evaluated
*afterwards* against the **last** trial's `${yuclid.@}` only. If two trials run for the
same point, metrics targeting the earlier one fail with `generated an empty string` and
the whole record is dropped. See "Trials as a dispatch table" below for how to keep them
mutually exclusive.

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
Unknown keys warn. Metric names become the column names consumed by `yuclid plot -y`.

A metric command may chain several pipelines with `;` to emit several numbers, which is
how per-region timers are collected in one column:

```json
{ "name": "stopwatches",
  "command": "grep Trial ${yuclid.@}.out | grep -oE '[0-9]+\\.[0-9]+' ; grep stopwatch ${yuclid.@}.out | sort | awk '{print $3}'" }
```

## `env`

```json
{ "env": { "root": "/my/path", "data_dir": "$root/data", "ALL_SIZES": "${yuclid.size.values}" } }
```

Each value is expanded by the shell (`echo "<value>"`) against the accumulated
environment, so later keys can reference earlier ones and any inherited variable.
Only global `${yuclid.*.values}` / `${yuclid.*.names}` are allowed here.

## `setup`

```json
{
  "setup": {
    "global": ["ulimit -s 1048576"],
    "point": [
      { "on": ["compiler"], "command": "mkdir -p ${yuclid.compiler}" },
      {
        "on": ["compiler"],
        "command": "make prog.out CXX=${yuclid.compiler} OUTDIR=$root/build/${yuclid.compiler}",
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
    { "command": ["OMP_NUM_THREADS=${yuclid.nthreads}", "burn_omp", "/usr/bin/time -v",
                  "$root/bin/${yuclid.program}_${yuclid.impl}.out", "${yuclid.input}", "2>&1"],
      "condition": "yuclid.program in ['bc', 'bfs', 'cc', 'pr']",
      "metrics": ["time", "space"] },
    { "command": ["OMP_NUM_THREADS=${yuclid.nthreads}", "burn_omp loop $nruns", "/usr/bin/time -v",
                  "$root/bin/${yuclid.program}_${yuclid.impl}.out", "${yuclid.input}", "2>&1"],
      "condition": "yuclid.program in ['canneal', 'streamcluster', 'xz']",
      "metrics": ["time", "space"] },
    { "command": ["LD_PRELOAD=$root/nmallocs/libnmallocs.so",
                  "$root/bin/${yuclid.program}_${yuclid.impl}.out", "${yuclid.input}"],
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
      "condition": "yuclid.program in ['bc', 'bfs', 'cc', 'pr']" },
    { "name": "time", "command": "egrep '\\[scoped_timer\\] Kernel: ' ${yuclid.@}.out | awk '{print $3}'",
      "condition": "yuclid.program in ['kmeans', 'xz']" },
    { "name": "time", "command": "egrep '\\[parsec\\] Total time spent in ROI: ' ${yuclid.@}.out | grep -oE '[0-9]+\\.[0-9]+'",
      "condition": "yuclid.program in ['canneal', 'streamcluster']" }
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

### Three tiers of variables

Long-lived configs use all three, and they resolve at different times:

| Form | Resolved | Use for |
|---|---|---|
| `${yuclid.dim}` | by yuclid, per point | anything that varies over the space |
| `$var` from `env` | by the shell, once at startup | derived paths (`"graphs_dir": "$root/gapbs/benchmark/graphs"`) |
| `$var` inherited | by the shell, from the caller's environment | machine/run identity — `$root`, `$machine`, `$nruns` |

Inherited variables never appear in the config's `env`; a driver script exports them and
then invokes `yuclid run`, which is how one config serves several machines:

```bash
export root=$(dirname $(realpath $0))
export machine="local"
export nruns="3"
yuclid run -i yuclid.json --select nthreads=28 impl=base cxx=clang++ "$@"
```

Keep that driver in the repo next to the config — it is the honest record of how a run was
parameterized, and `--select`/`--metrics` lines can be commented in and out per experiment.

### Deferring expansion to an inner shell

`\\$VAR` in JSON reaches the shell as `\$VAR`, which survives yuclid's shell as the literal
text `$VAR`. Use it when a wrapper re-evaluates the command (`loop`/`burn_omp` end in
`eval "$cmd"`) and the variable is only set at that inner level:

```json
{ "name": "large", "value": "10 20 128 1000000 200000 5000 none out.txt \\$OMP_NUM_THREADS" }
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
      { "on": ["cxx"], "command": "mkdir -p $root/.yuclid/bin/$machine/${yuclid.cxx}", "parallel": true },
      { "on": ["cxx", "program", "impl"],
        "command": "make ${yuclid.program}_${yuclid.impl} OUTDIR=$root/.yuclid/bin/$machine/${yuclid.cxx} CXX=${yuclid.cxx}",
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

`-i` merges: `env`, `space`, and `presets` merge as dicts; `trials`, `metrics`, and `order`
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
- **Multiple inputs merge.** `yuclid run -i base.json local.json` merges `env`, `space`,
  and `presets` as dicts (later wins per key) and concatenates `trials`, `metrics`, and
  `order` — useful for machine-specific overlays.
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
  the grep targeted the wrong file (`.out` vs `.err`) or a second trial overwrote
  `${yuclid.@}`.
- **`${yuclid.@}` in a metric is the last trial's id**, not the id of the trial that
  declared the metric. Keep one trial per point per metric group.
- Trial `condition`s see values, so `yuclid.program in ['bc', 'bfs']` works on a plain
  string dimension but a list membership test against a `null` dimension compares strings.
- `${yuclid.dim}` yields the *value*; use `${yuclid.dim.name}` for the label. Output
  records carry names, not values.

## Verify before handing off

Always dry-run. It parses and normalizes the config, prints the `env` bindings, the
resolved subspace per dimension, its total size, and every point it would visit in order —
without executing anything:

```
yuclid run --dry-run
yuclid run --dry-run -s size=small cpuid=0     # shrink the space first
yuclid run -s size=small                       # then a real smoke run
```

Then check the JSONL and plot it:

```
yuclid tplot results.jsonl -x size -z compression -y time.real -A
```

Other `yuclid run` flags worth mentioning: `-o/--output`, `--output-dir`, `-p` (presets),
`-m` (subset of metrics), `-r N` (repeat each point), `--parallel-trials [N]`, `--fold`,
`--no-setup`, `--temp-dir`.

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

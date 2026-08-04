# Configuration

Yuclid builds the Cartesian product of the dimensions in `space`, runs the
compatible `trials` at each point, evaluates `metrics`, and writes one record
per point. Start with `space`, `trials`, and `metrics`; add setup, environment,
presets, and ordering only when the experiment needs them.

## Files and formats

The default configuration is the first of `yuclid.json`, `yuclid.yaml`, or
`yuclid.yml` found in the current directory. Name other files with `-i`:

```sh
yuclid run -i base.json machine.yaml
```

JSON and YAML have the same structure. YAML support requires
`pip install 'yuclid[yaml]'`.

When several files are supplied, `space` and `presets` are merged by key;
`env`, `trials`, `metrics`, `setup`, and `order` are appended in file order.
Later definitions of the same space or preset key replace earlier ones.

Exactly seven top-level fields are recognized:

| Field | Purpose |
|---|---|
| `space` | Dimensions and their values |
| `trials` | Experiment commands |
| `metrics` | Commands that extract numeric results |
| `env` | Environment variables for commands |
| `setup` | Global and point-specific preparation |
| `presets` | Named subspaces |
| `order` | Dimension iteration order |

`trials` must contain at least one valid command. `metrics` may be omitted,
but the run will collect no measurements.

## Complete example

```json
{
  "env": {
    "build_dir": "build"
  },
  "space": {
    "size": [1000, 10000],
    "compiler": [
      { "name": "gcc", "value": "gcc" },
      { "name": "clang", "value": "clang" }
    ],
    "optimization": ["O1", "O3"]
  },
  "setup": {
    "point": {
      "on": ["compiler", "optimization"],
      "command": "${yuclid.compiler} -${yuclid.optimization} benchmark.c -o $build_dir/${yuclid.compiler.name}-${yuclid.optimization}"
    }
  },
  "trials": "$build_dir/${yuclid.compiler.name}-${yuclid.optimization} ${yuclid.size}",
  "metrics": {
    "seconds": "grep -oE '[0-9]+\\.[0-9]+' ${yuclid.@}.out"
  },
  "presets": {
    "quick": { "size": [1000], "compiler": ["gcc"] }
  },
  "order": ["compiler", "optimization", "size"]
}
```

Inspect it without executing commands:

```sh
yuclid run --dry-run
yuclid run -p quick
```

## Space

`space` maps each dimension name to its possible points. Dimension names may
contain letters, digits, and underscores.

Scalar points use the same value in commands and results:

```json
{ "space": { "threads": [1, 2, 4], "mode": ["fast", "safe"] } }
```

Object points separate the readable `name` stored in results from the `value`
substituted into commands:

```json
{
  "space": {
    "dataset": [
      { "name": "small", "value": "data/input-10M.bin" },
      { "name": "large", "value": "data/input-1G.bin" }
    ]
  }
}
```

A point object requires `value` and may also contain:

- `name`: result label; defaults to the string form of `value`.
- `condition`: Python expression deciding whether the point exists.
- `setup`: command or list of commands associated with that value.

Conditions refer to other dimension values through `yuclid.<dimension>`:

```json
{
  "space": {
    "backend": ["cpu", "gpu"],
    "device": [
      { "name": "host", "value": 0, "condition": "yuclid.backend == 'cpu'" },
      { "name": "cuda0", "value": 0, "condition": "yuclid.backend == 'gpu'" }
    ]
  }
}
```

Conditions are Python expressions, not shell expressions. They preserve JSON
types, so numeric space values compare as numbers.

An undefined dimension is written as `null` and must be supplied by a selector
or preset:

```json
{ "space": { "machine": null } }
```

```sh
yuclid run --select machine=laptop,server
```

Selector values for undefined dimensions are strings. Cast them in numeric
conditions, for example `int(yuclid.threads) > 1`.

A generated dimension uses a `:py` suffix. Its value is evaluated as Python at
load time and must produce a list:

```json
{ "space": { "threads:py": "list(range(1, 9))" } }
```

## Commands and substitutions

A command may be a string or a list of strings; lists are joined with spaces.
Normal shell variables remain available.

Point-scoped commands (`trials`, `metrics`, `setup.point`) support:

| Form | Meaning |
|---|---|
| `${yuclid.dim}` | Current point value |
| `${yuclid.dim.value}` | Current point value |
| `${yuclid.dim.name}` | Current point name |
| `${yuclid.@}` | Unique capture path for the current trial |

Yuclid captures every trial's standard output and error as
`${yuclid.@}.out` and `${yuclid.@}.err`; trials do not need to redirect them.

Global contexts (`env` and `setup.global`) have no current point and support:

| Form | Meaning |
|---|---|
| `${yuclid.dim.values}` | All selected values, space-separated |
| `${yuclid.dim.names}` | All selected names, space-separated |

Point variables are invalid in global scope, and global list variables are
invalid in point scope.

## Trials

`trials` is one command or a list of commands and trial objects:

```json
{
  "trials": [
    "./benchmark ${yuclid.size}",
    {
      "command": "perf stat ./benchmark ${yuclid.size}",
      "condition": "yuclid.mode == 'profile'",
      "metrics": ["cache_misses"]
    }
  ]
}
```

A trial object has:

- `command` (required): string or list of strings.
- `condition`: Python expression over the current point; default `True`.
- `metrics`: metric names this trial enables; omitted means all compatible
  metrics.

When a metric reads `${yuclid.@}`, exactly one compatible trial may enable that
metric at a point. Use mutually exclusive trial conditions or disjoint metric
lists when different regions need different commands.

## Metrics

A metric command reads trial output and must print one or more numbers separated
by whitespace. Multiple values are averaged by default or retained as an array
with `yuclid run --fold`.

The concise mapping form is suitable when no metric conditions are needed:

```json
{
  "metrics": {
    "seconds": "grep elapsed ${yuclid.@}.out | awk '{print $2}'",
    "bytes": "grep bytes ${yuclid.@}.out | awk '{print $2}'"
  }
}
```

The list form supports conditions:

```json
{
  "metrics": [
    {
      "name": "seconds",
      "command": "grep elapsed ${yuclid.@}.out | awk '{print $2}'",
      "condition": "yuclid.backend == 'cpu'"
    }
  ]
}
```

`name` and `command` are required. An empty, nonnumeric, or failing metric
marks the point failed and prevents its record from being written. Select a
subset with `yuclid run --metrics seconds bytes`.

## Environment

`env` may be one object or a list of objects applied in order:

```json
{
  "env": [
    { "root": "/opt/bench" },
    { "data": "$root/data", "PATH": "$root/bin:$PATH" }
  ]
}
```

Entries in one object are an independent group and cannot refer to each other.
Put dependencies in a later object. `$NAME`, `${NAME}`, and inherited
environment variables are expanded; `$$` produces a literal dollar.

## Setup

`setup.global` runs before the experiment and has global substitution scope:

```json
{ "setup": { "global": ["mkdir -p $build_dir", "make data"] } }
```

`setup.point` runs once per combination of selected dimensions rather than once
per full experiment point:

```json
{
  "setup": {
    "point": [
      {
        "on": ["compiler", "optimization"],
        "command": "make CC=${yuclid.compiler} OPT=-${yuclid.optimization}",
        "parallel": ["compiler"]
      }
    ]
  }
}
```

A point setup item has:

- `command` (required): string or list of strings.
- `on`: dimensions whose combinations receive the command; omitted means all.
  Use `dim.names` instead of the default `dim.values` when the name is needed.
- `parallel`: `false`, `true` for every `on` dimension, or a list of `on`
  dimensions that may run concurrently.

`--no-setup` skips global and point setup.

## Presets and selectors

Presets name reusable subspaces. Entries select point names, and `*` is a glob:

```json
{
  "presets": {
    "quick": { "size": ["small"], "compiler": ["clang*"] },
    "scaling": { "threads": [1, 2, 4, 8] }
  }
}
```

```sh
yuclid run --presets quick
yuclid run --select size=small,medium compiler=gcc
```

Presets may define literal values for a `null` dimension, but globs cannot be
used there. Selectors restrict defined dimensions by point name.

## Order

`order` controls traversal. Listed dimensions are moved to the end in the order
given, and the last varies fastest:

```json
{ "order": ["compiler", "dataset", "threads"] }
```

Here all thread counts run consecutively for one dataset and compiler. The CLI
`--order` option applies after the configured order.

## Output and validation

Each successful point produces one JSON Lines or CSV record containing its
dimension names and metrics:

```json
{"size":"small","compiler":"gcc","seconds":0.37}
```

Use a dry run to inspect the resolved environment, subspace, commands, and
point order before executing anything:

```sh
yuclid run --dry-run
yuclid run --dry-run --select size=small
yuclid run --select size=small -o smoke.yuclid.jsonl
```

JSON is strict: comments and trailing commas are invalid, and backslashes in
regular expressions must be escaped. See the runnable
[`examples`](../examples/README.md) for progressively larger configurations.

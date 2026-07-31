# Yuclid

*Combinatorially explode your experiments*

<p><img src="space.png" align="right" width="350" height="298"/></p>

Yuclid is a tool for orchestrating experiments in N-dimensional irregular spaces of parameters.
It collects custom metrics in a single JSON file for easy post-processing.
Yuclid builds the Cartesian product of the dimensions you defined, and runs an experiment per point in that space.
It also provides a unique way of plotting data (`yuclid plot`) interactively, browsing slices of the results using the arrow keys.

The **geometrical metaphor** is that each experiment is a point in a multidimensional discrete space formed by all combinations of user-defined parameters.
By specifying extra conditions (see advanced example), some hyper regions can be carved out of the original space.

## What kind of experiments?

Anything that can be expressed in a single (pipelined) command that generates one or more numbers.
Since programs' outputs are often verbose and the target metric is contained in a single line,
metrics can be arbitrarily defined in terms of other commands, e.g., regular expressions (see example).

Here's a list of use-case ideas:
- Measure the impact of different optimization levels of different **compilers** on different programs
- Count cache misses under different **memory allocators** on different inputs
- Measure strong scaling **parallel programs** given different thread affinities
- Evaluate different compression algorithms on different inputs with different compression levels
- Organize **perf** counters alongside custom metrics e.g., max RSS, in a self-contained JSON file
- Create reproducible artifacts for **research** software
- All of the above combined!

## Installation

Requires python >= 3.10

Development head:
```
pip install git+https://github.com/fsossai/yuclid.git
```

Stable release:
```
pip install yuclid
```

- **`yuclid run`**: Run experiments with all combinations of the defined parameters.
- **`yuclid describe`**: Report what a result file holds, and which combinations it does not.
- **`yuclid plot`**: Interactively visualizes the results produced by `yuclid run`.

## Configuration for `yuclid run`

The configuration may be written as JSON (`yuclid.json`) or as YAML
(`yuclid.yaml`), whichever you prefer; `yuclid run` picks up either. The two
forms are interchangeable, so everything below applies to both. YAML needs
PyYAML: `pip install yuclid[yaml]`.

Key sections:
- **`env`**: Environment variables and constants
- **`setup`**: Commands to run before experiments (`global`) or for specific parameter combinations (`point`)
- **`trials`**: The actual experiment commands that generate metrics to collect
- **`metrics`**: How to extract a given metric from the data collected by the trials
- **`space`**: Dimension definitions - all combinations will be explored
- **`order`**: Execution order of parameter combinations

Parameters can be simple lists or objects with `name`/`value` pairs.
Use `${yuclid.x}` in a command to reference the value of dimension `x`, and `${yuclid.@}` for a unique output filename.
`${yuclid.x}` is an alias for `${yuclid.x.value}`.

A value may also carry a **condition**, a boolean expression over the point that
decides whether that point exists at all — this is what makes the space
irregular. A dimension declared with no values is **undefined**, and has to be
supplied at invocation with `--select nthreads=1,7,14`. A **preset** is a named
subspace, run with `-p`.

The examples below show all of it in place.

## Examples

Five worked experiments live in [`examples/`](examples/), each runnable as it
stands and each written to teach one part of yuclid:

| Example | Goal |
|---|---|
| [matrix-multiplication](examples/matrix-multiplication/) | A plain space, point setup on one dimension, and presets |
| [cache-traversal](examples/cache-traversal/) | Conditions, several metrics from one trial, and wrapping a workload |
| [compression-codecs](examples/compression-codecs/) | Point setup shared across a subspace, and metrics that are not timings |
| [sqlite-ingestion](examples/sqlite-ingestion/) | Conditions over other dimensions, and a private file per point |
| [parallel-prime-sieve](examples/parallel-prime-sieve/) | Repetitions, and what to do about a noisy measurement |

Each directory holds a `yuclid.json` with an equivalent `yuclid.yaml`, the
workload it measures, no third-party dependencies, and a README pointing at the
parts of the configuration worth reading. Read them in that order if you are
new to yuclid.

```sh
cd examples/matrix-multiplication
yuclid run --dry-run   # print every command, run none of them
yuclid run -p quick    # a small corner of the space
```

A subspace can also be selected on the command line:

```sh
yuclid run -s size=medium
yuclid run -s cpuid=0,1,2
yuclid run -s size=small,medium cpuid=3,0
```

## What a run produces

One JSON Lines record per point, holding the name of every dimension and the
metrics collected there:

```json
{"size": "small", "cpuid": "0", "compression": "lowest", "time.real": 0.37, "time.sys": 0.05}
{"size": "small", "cpuid": "0", "compression": "highest", "time.real": 0.33, "time.sys": 0.05}
{"size": "medium", "cpuid": "0", "compression": "lowest", "time.real": 0.62, "time.sys": 0.11}
{"size": "large", "cpuid": "0", "compression": "lowest", "time.real": 1.59, "time.sys": 0.27}
```

`yuclid describe` reports what such a file holds: the dimensions with their
values, the metrics with their range, and the combinations that are absent.

The results can be displayed in a window:

```
yuclid plot results.yuclid.jsonl -x compression
yuclid plot results.yuclid.jsonl -x size -z cpuid
```

Interact with the plot using arrow keys to move around dimensions and number keys to change the metric!

Or entirely in the terminal with `yuclid tplot`:
```
yuclid tplot results.yuclid.jsonl -x size -z compression -y time.real -A
```
```
                                    [1] time.real                               
    ┌──────────────────────────────────────────────────────────────────────────┐
1.59┤###########  @@@@@@@@@@@                             ## highest           │
    │###########  @@@@@@@@@@@                              @@ lowest           │
    │###########  @@@@@@@@@@@                                                  │
1.33┤###########  @@@@@@@@@@@                                                  │
    │###########  @@@@@@@@@@@                                                  │
1.06┤###########  @@@@@@@@@@@                                                  │
    │###########  @@@@@@@@@@@                                                  │
    │###########  @@@@@@@@@@@                                                  │
0.80┤###########  @@@@@@@@@@@ ###########                                      │
    │###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@                         │
0.53┤###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@                         │
    │###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@              @@@@@@@@@@@│
0.27┤###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@│
    │###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@│
0.00┤###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@ ###########  @@@@@@@@@@@│
    └───────────┬─────────────────────────┬────────────────────────┬───────────┘
              large                    medium                    small          
time.real                               size

        large   medium  small
highest 1.58    0.66    0.33
lowest  1.59    0.62    0.37
```

## Output formats

Results are JSON Lines by default. `--format csv` writes a CSV instead, with
one header naming every dimension and metric:

```
yuclid run --format csv
yuclid run -o results.csv          # the extension decides on its own
```

`yuclid plot`, `yuclid tplot` and `yuclid stats` read either, chosen by the
file's extension. A CSV carries no types, so a dimension whose values are
numbers is indistinguishable from a metric: name the metrics with `-y` when
reading one.

`--fold` has no CSV form, since a cell holds one value rather than an array of
samples.

## Reproducible scripts

`yuclid run --compile experiment.sh` writes a shell script instead of running
anything. Every point of the space is unrolled, so the script contains no loops
and no branches — just the commands, in the order yuclid would have run them:

```sh
yuclid run -p quick --compile experiment.sh
sh experiment.sh                    # no yuclid, no configuration needed
```

It reproduces the same JSON Lines, appended to the same place, and captures
each trial's output the same way. `YUCLID_OUTPUT` and `YUCLID_WORK` override
the two destinations. Selectors, presets, conditions and `--repeat` are all
resolved while compiling, so the script is a record of exactly one experiment.

Numbers keep the formatting the measured program printed, rather than being
re-serialized.

## Skills

If you use Claude, symlink [`skills/`](skills/) into your agent directory and
it will pick both of them up:

```sh
ln -s "$PWD/skills" ~/.claude/skills/yuclid       # everywhere
ln -s "$PWD/skills" .claude/skills/yuclid         # this project only
```

- **`yuclid-json`** writes and fixes a configuration: the space, the trials,
  and the commands that scrape the numbers out of them.
- **`yuclid-plot`** reads a result file and suggests what is worth looking at,
  as `yuclid plot` / `tplot` / `stats` commands you can paste.

## Plot API

`yuclid plot` can be used directly on your pyplot canvas. The command `yuclid plot results.yuclid.jsonl -x size -z cpuid` can be emulated in a more customizable script, e.g.:

```python
import yuclid.plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# just like the CLI
cli_args = [
  "results.yuclid.jsonl",
  "-x",
  "size",
  "-z",
  "cpuid"
]
df = yuclid.plot.draw(fig, ax, cli_args)
plt.show()
```

## Tests

`yuclid run` has a black-box test suite. No dependencies beyond the standard library:

```
python tests/run_tests.py            # run everything
python tests/run_tests.py -k preset  # only cases whose name contains "preset"
```

See [tests/README.md](tests/README.md) for how to add a case.

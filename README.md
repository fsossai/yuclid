# Yuclid

*Combinatorially explode your experiments*

<p><img src="space.png" align="right" width="350" height="298"/></p>

Yuclid is a tool for orchestrating experiments in N-dimensional irregular spaces of parameters.
It collects custom metrics in a single JSON file for easy post-processing.
Yuclid builds the Cartesian product of the dimensions you defined, and runs an experiment per point in that space.
It also provides a unique way of plotting data (`yuclid plot`) interactively, browsing slices of the results using the arrow keys.

The **geometrical metaphor** is that each experiment is a point in a multidimensional discrete space formed by all combinations of user-defined parameters.
By specifying extra conditions, some hyper regions can be carved out of the original space.

## What kind of experiments?

Anything that can be expressed in a single (pipelined) command that generates one or more numbers.
Since programs' outputs are often verbose and the target metric is contained in a single line,
metrics can be arbitrarily defined in terms of other commands, e.g., regular expressions.

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

Check out the [`examples`](examples/README.md)!

## What a run produces

One CSV or JSON Lines record per point, holding the name of every dimension and the
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
  as `yuclid plot` / `tplot` / `stats` commands that you can paste.


## Reproducible scripts

`yuclid run --compile experiment.sh` writes a shell script instead of running
anything. Every point of the space is unrolled, so the script contains no loops
and no branches — just the commands, in the order yuclid would have run them:

```sh
yuclid run -p quick --compile experiment.sh
sh experiment.sh                    # no yuclid, no configuration needed
```

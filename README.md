# Yuclid

*Combinatorially explode your experiments*

Yuclid is a command-line tool with a Web UI for orchestrating and visualizing
experiments across irregular, N-dimensional parameter spaces.
It collects custom metrics in a single JSONL file for easy post-processing.
Yuclid builds the Cartesian product of the dimensions you define and runs an experiment at each point in that space.
Experiments can be monitored and steered in real-time via CLI or Web UI.

Check out the [examples](examples/README.md).

![Watching a run of the mandelbrot example in the Web UI](https://raw.githubusercontent.com/fsossai/yuclid/master/mandelbrot.gif)

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

## Overview

The main workflow is to run an experiment, watch or steer it while it is in
progress, then inspect and visualize the resulting dataset.

### Run and monitor experiments

- **`yuclid run`** builds the configured space, executes every selected point,
  and writes its dimensions and metrics to JSON Lines or CSV.
- **`yuclid serve`** opens the browser interface for watching, launching, and
  steering runs recorded in the current directory.
- **`yuclid status`** reports the progress of a live run; `--watch` keeps the
  report current until it ends.
- **`yuclid runs`** lists recent recorded runs and their result files.

### Steer a live run

- **`yuclid pause`** and **`yuclid resume`** temporarily stop and continue
  scheduling work.
- **`yuclid drop`** removes matching points from the remaining plan;
  **`yuclid add`** restores them or extends the plan with new values.
- **`yuclid repeat`** changes the repetition count for remaining points, and
  **`yuclid order`** changes their traversal order.
- **`yuclid kill`** abandons a point or repetition currently in flight;
  **`yuclid stop`** ends the run and interrupts its active commands.

### Revisit recorded runs

- **`yuclid finish`** measures the points a previous run left without results.
- **`yuclid replay`** optimistically runs a previous run again, preserving its
  intended points and repetitions while retrying failures.

### Inspect and visualize results

- **`yuclid describe`** summarizes a result file and reports missing points.
- **`yuclid plot`** explores slices of a dataset in a graphical interface.
- **`yuclid tplot`** provides the interactive plotter in a terminal.
- **`yuclid stats`** plots the distribution of a selected metric.

A plot puts one dimension on the horizontal axis, uses another for the series,
and leaves the rest free. The arrow keys walk through those, so one command is
a whole family of plots rather than a single picture — which is what the two
animations below are stepping through:

```sh
yuclid plot results.jsonl -x threads -z schedule -y seconds -R threads=1 schedule=static compiler=gcc -r -l -A
```

![Speedup by thread count and schedule, stepping through compiler and image size](https://raw.githubusercontent.com/fsossai/yuclid/master/examples/mandelbrot/plot.gif)

```sh
yuclid plot results.jsonl -x threads -z compiler -y seconds -A
```

![Seconds by thread count and compiler, stepping through schedule and image size](https://raw.githubusercontent.com/fsossai/yuclid/master/examples/mandelbrot/bars.gif)

### Agent integration

- **`yuclid skills`** installs or uninstalls Yuclid's configuration and plotting
  skills for Codex, Claude, or a custom Agent Skills directory.

Run `yuclid <command> --help` for the complete options of any subcommand.

## Configuration

See [Configuration](docs/configuration.md).

## Skills

Yuclid includes skills for agents that support the Agent Skills format. A few
examples of how to install them:

```sh
yuclid skills install --agent codex                 # Codex, user-wide
yuclid skills install --agent claude                # Claude, user-wide
yuclid skills install --directory .agents/skills    # custom or project directory
```

- **[`yuclid-config`](yuclid/agent_skills/yuclid-config/SKILL.md)** writes and fixes a configuration: the space, the trials,
  and the commands that extract metrics from their output.
- **[`yuclid-plot`](yuclid/agent_skills/yuclid-plot/SKILL.md)** reads a result file, suggests useful views, and produces
  ready-to-run `yuclid plot`, `tplot`, and `stats` commands.


## Reproducible scripts

`yuclid run --compile experiment.sh` writes a shell script instead of running
anything. Every point of the space is unrolled, so the script contains no loops
and no branches — just the commands, in the order yuclid would have run them:

```sh
yuclid run -p quick --compile experiment.sh
sh experiment.sh                    # no yuclid, no configuration needed
```

# Yuclid

*Combinatorially explode your experiments*

<p><img src="space.png" align="right" width="350" height="298"/></p>

Yuclid is a CLI + Web UI tool for orchestrating and visualizing experiments in N-dimensional irregular spaces of parameters.
It collects custom metrics in a single JSONL file for easy post-processing.
Yuclid builds the Cartesian product of the dimensions you defined, and runs an experiment per point in that space.
Experiments can be monitored and steered in real-time via CLI or Web UI.

Check out the [examples](examples/README.md).

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
  and the commands that scrape the numbers out of them.
- **[`yuclid-plot`](yuclid/agent_skills/yuclid-plot/SKILL.md)** reads a result file and suggests what is worth looking at,
  as `yuclid plot` / `tplot` / `stats` commands that you can paste.


## Reproducible scripts

`yuclid run --compile experiment.sh` writes a shell script instead of running
anything. Every point of the space is unrolled, so the script contains no loops
and no branches — just the commands, in the order yuclid would have run them:

```sh
yuclid run -p quick --compile experiment.sh
sh experiment.sh                    # no yuclid, no configuration needed
```

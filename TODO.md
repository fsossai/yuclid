# TODO

## Quick

- add: option to run the `serve` in the background
- add: stddev column when comparing "selected" and "other" in the space
- fix: make sure cmd line options are consistent with what they do, especially `--workspace`
- fix: "export run" should export the file results.yuclid.jsonl to the cwd

## Planned

- add: alternative execution order that prioritizes exploration of the space rather then repetitions
- add: option to send an email upon completion
- add: real-time plotter inside web UI with x/y/z parameters composable via buttons
- add: turn a hyperfine command into a yuclid configuration
- add: turn a yuclid command into a hyperfine configuration
- add: workspace options accessible in the UI including: update frequency, enable/disable the creation of JSONL files in the CWD, preferred log order, preferred space nesting order, preferred comparison average (mean, median)
- add: comparison with hyperfine under docs/
- change: `--fold` pads the shorter metrics of a point with `NaN`
- change: single web server with workspace manager
- change: move the space panel into the run panel
- change: support only one configuration file per workspace
- change: readme image
- research: find hyperfine users that extensively use parameter lists
- research: how yuclid compares to hyperfine

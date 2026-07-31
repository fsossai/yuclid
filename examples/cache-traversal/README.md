# Cache and memory traversal

Imagine having to measure how fast memory is when you read it in different
ways: straight through, skipping ahead, jumping around, or following pointers.
Each way has to be tried on arrays of several sizes, and some of them take a
stride while others do not.

This example shows how to describe a space that is not a plain grid, and how to
collect several numbers from one run.

## What to look at in `yuclid.json`

**Conditions.** `stride` applies only to the `strided` pattern. Each value
carries a `condition`, and yuclid drops the points where it does not hold. The
space becomes irregular: there is no `linear` point with a 4 KiB stride,
because it would mean nothing.

**Many metrics from one trial.** The trial runs once and prints eight numbers
on one line. Each metric picks its own column out of `${yuclid.@}.out`, the
file where yuclid captured that trial's output.

**A wrapper around the real program.** `measure.py` adds `perf` counters when
they are available and zeroes when they are not, reporting which case it was in
through `perf_available`. Wrapping the workload in a small script is often
easier than making the configuration handle every machine.

**A second preset.** `memory-pressure` selects the two largest sizes. Presets
are named subspaces: you can keep several ways of running the same experiment
in one file.

Needs a C11 compiler as `cc`.

## Running it

```sh
yuclid run --dry-run
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x mebibytes -z pattern -y seconds -f stride=none
yuclid tplot yuclid.results.jsonl -x mebibytes -z stride -y seconds -f pattern=strided
```

Add `-r 5` to repeat every point five times, then look at the distributions:

```sh
yuclid stats yuclid.results.jsonl -y seconds -z pattern
```

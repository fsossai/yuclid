# Cache and memory traversal

Imagine having to measure how fast memory is when you read it in different
ways: straight through, skipping ahead, jumping around, or following pointers.
Each way has to be tried on arrays of several sizes, and only one of them takes
a stride.

## What to look at in `yuclid.json`

**Conditions.** `stride` applies only to the `strided` pattern. Each value of
`stride` carries a `condition`, and yuclid skips the points where it does not
hold. The space is then not a full product: there is no `linear` point with a
4 KiB stride.

**Several metrics from one trial.** The trial runs once and prints eight
numbers on one line. Each metric picks one column out of `${yuclid.@}.out`,
which is the file where yuclid captured that trial's standard output.

**A wrapper around the program.** The trial calls `measure.py`, which runs the
C program under `perf` when it is available and reports zeros otherwise, saying
which case it was in through `perf_available`. The configuration stays the same
on every machine.

**Two presets.** `quick` selects the two smallest sizes, `memory-pressure` the
two largest. Several ways of running the same experiment can live in one file.

Needs a C11 compiler as `cc`.

## Running it

```sh
yuclid run --dry-run
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x mebibytes -z pattern -y seconds -f stride=none
yuclid tplot yuclid.results.jsonl -x mebibytes -z stride -y seconds -f pattern=strided
```

With `-r 5` each point is run five times, and `yuclid stats` shows the
distribution of the samples:

```sh
yuclid run -p quick -r 5 -o yuclid.results.jsonl
yuclid stats yuclid.results.jsonl -y seconds -z pattern
```

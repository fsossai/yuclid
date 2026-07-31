# Compression codecs

Imagine having to pick a compression codec. There are three to compare, at
three levels each, on three kinds of data of different sizes — and the answer
depends on whether you care about time, size, or memory.

This example shows how to build the input files once and reuse them across
every combination that needs them.

## What to look at in `yuclid.json`

**Point setup on part of the space.** A corpus depends on `kind` and
`mebibytes`, not on the codec or the level. `on: ["kind", "mebibytes"]` builds
each corpus once and lets all nine codec/level runs share it. Without that
list, the same file would be written nine times.

**Substitutions in a filename.** The corpus path is built from the point
itself, `data/${yuclid.kind}-${yuclid.mebibytes}MiB.bin`, so setup and trial
agree on where the file is without repeating a rule.

**Metrics that are not timings.** `ratio` and `compressed_bytes` are sizes,
`round_trip_ok` is 1 or 0. Any command that prints a number can be a metric,
which is a cheap way to record whether a run should be trusted at all.

**Plain lists.** `kind`, `codec` and `level` are written as bare lists. When
the value is already readable, there is no reason to give it a separate name.

## Running it

```sh
yuclid run --dry-run
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x level -z codec -y compression_seconds -f kind=text
yuclid tplot yuclid.results.jsonl -x codec -z kind -y ratio -f level=6
```

Press the number keys to switch between metrics inside the viewer.

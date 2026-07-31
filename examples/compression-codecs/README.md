# Compression codecs

Imagine having to pick a compression codec. There are three to compare, at
three levels each, on three kinds of data of different sizes. Both directions
have to be timed, and the compressed size matters as much as the time.

## What to look at in `yuclid.json`

**Point setup on part of the space.** A corpus depends on `kind` and
`mebibytes`, not on the codec or the level. `on: ["kind", "mebibytes"]` builds
each corpus once, and the nine codec/level combinations that need it share the
file. Without that list the setup command would run for every point.

**Substitutions in a filename.** The corpus path is built from the point:
`data/${yuclid.kind}-${yuclid.mebibytes}MiB.bin`. Setup and trial derive the
same path from the same dimensions.

**Metrics that are not timings.** `ratio` and `compressed_bytes` are sizes, and
`round_trip_ok` is 1 or 0. Any command that prints a number can be a metric.

**Plain lists.** `kind`, `codec` and `level` are written as bare lists, so each
value is its own name.

## Running it

```sh
yuclid run --dry-run
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x level -z codec -y compression_seconds -f kind=text
yuclid tplot yuclid.results.jsonl -x codec -z kind -y ratio -f level=6
```

The number keys switch between metrics inside the viewer.

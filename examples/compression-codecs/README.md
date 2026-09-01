# Compression codecs

Imagine having to pick a compression codec. There are three to compare at
user-selected levels, across three types and sizes of data. Compression and
decompression have to be timed, and the compressed size matters as much as the
time.

## The Space

A corpus depends on `kind` and `mebibytes`, not on the codec or the level. `on: ["kind", "mebibytes"]` builds
each corpus once, and all codec/level combinations that need it share the file.
Without that list the setup command would run for every point.

The dimension `level` is undefined (`null`). This means that its values must be chosen by the user when the
run starts, for example, `-s level=1,6`.
Values added during steering are preserved when the run is replayed.

The metrics cover compression and decompression time, compressed size,
compression ratio, and round-trip correctness. Any command that prints a
number can be a metric.


## Running it

```sh
yuclid run --dry-run -s level=1,6,9
yuclid run -p quick -s level=1,6  # this will produce a file like 20260731-120000.yuclid.jsonl

yuclid tplot 20260731-120000.yuclid.jsonl -x level -z codec -y compression_seconds -f kind=text
yuclid tplot 20260731-120000.yuclid.jsonl -x codec -z kind -y ratio -f level=6
```

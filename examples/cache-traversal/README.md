# Cache and memory traversal

One array is walked four ways — `linear` contiguous reads, `strided` reads,
deterministic `random` reads, and `pointer` chasing around a shuffled cycle —
at working-set sizes from 8 MiB to 512 MiB, which crosses the cache hierarchy
of a typical machine.

`stride` applies only to the `strided` pattern and is `none` everywhere else;
its values are named by the distance they cover in bytes.

`measure.py` wraps the C program. On Linux with `perf` on the path it adds
cycle, instruction, cache-miss and branch-miss counters; elsewhere those four
metrics are 0 and `perf_available` is 0. Filter on `perf_available=1` before
reading anything into them.

Needs a C11 compiler as `cc`.

```sh
yuclid run -p quick -o results.jsonl

yuclid tplot results.jsonl -x mebibytes -z pattern -y seconds -f stride=none
yuclid tplot results.jsonl -x mebibytes -z stride -y seconds -f pattern=strided
yuclid stats results.jsonl -y seconds -z pattern
```

The `memory-pressure` preset walks 128 and 512 MiB, well past any cache. Add
`-r 5` to turn each point into a distribution worth looking at with
`yuclid stats`.

# Parallel segmented prime sieve

A segmented Sieve of Eratosthenes, varying the upper limit, the number of
worker threads, how segments are handed out (`static` split versus `dynamic`
stealing through an atomic counter) and the segment size, which trades
scheduling overhead against cache footprint.

`prime_count` is fixed for a given limit, so it doubles as a correctness check
across every scheduling choice. There are no generated inputs: the workload is
its own input.

Needs a C++17 compiler as `c++`.

```sh
yuclid run -p quick -r 3 -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x workers -z schedule -y tested_per_second \
  -f segment_kib=64KiB
yuclid tplot yuclid.results.jsonl -x segment_kib -z workers -y peak_rss_kib \
  -f schedule=dynamic
```

`-r 3` matters here: at 1 million the run is short enough that a single
measurement is mostly timer noise. The `scaling` preset goes to 20 million,
where thread scaling rises clearly above it.

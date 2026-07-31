# Parallel prime sieve

Imagine having to tune a parallel program. There is a thread count to choose, a
way of handing work out, and a chunk size — and the run is short enough that
one measurement tells you nothing.

This example is the simplest configuration of the five, and it shows what to do
about noise.

## What to look at in `yuclid.json`

**A space and nothing else.** No setup beyond compiling, no generated inputs:
the workload is its own input. Four dimensions, four metrics, one trial.

**Repetitions.** `yuclid run -r 5` runs every point five times and writes one
record per run. Nothing in the configuration changes; the results simply carry
five samples per point, and the viewers summarise them with a median and a
spread.

**Values that are already names.** `workers` is the list `[1, 2, 4, 8]`. When a
value has no better name, use it as it is.

**A metric that is not a measurement.** `prime_count` is the same for a given
limit, whatever the thread count. Recording it costs nothing and shows
immediately if a scheduling choice broke the answer.

Needs a C++17 compiler as `c++`.

## Running it

```sh
yuclid run --dry-run
yuclid run -p quick -r 5 -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x workers -z schedule -y tested_per_second \
  -f segment_kib=64KiB
yuclid tplot yuclid.results.jsonl -x segment_kib -z workers -y peak_rss_kib \
  -f schedule=dynamic
```

`-m sd,2` draws two standard deviations around each point instead of the
default percentile interval.

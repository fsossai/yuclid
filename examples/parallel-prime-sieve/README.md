# Parallel prime sieve

Imagine having to tune a parallel program. There is a thread count to choose, a
way of handing work out, and a chunk size. The run is short, so a single
measurement of it varies from one execution to the next.

## What to look at in `yuclid.json`

**A space and nothing else.** The only setup is the compilation, and there are
no generated inputs. Four dimensions, one trial, four metrics.

**Repetitions.** `-r 5` runs every point five times and writes one record per
run. The configuration is unchanged; the results carry five samples per point,
and the viewers reduce them to a median with a spread around it.

**Values that are their own names.** `workers` is the list `[1, 2, 4, 8]`,
while `limit` and `segment_kib` use `name`/`value` pairs to keep long numbers
readable in the plots.

**A metric that is not a measurement.** `prime_count` depends only on `limit`,
so it stays constant across the scheduling choices and shows when one of them
produced a different answer.

Needs a C++17 compiler as `c++`.

## Running it

```sh
yuclid run --dry-run
yuclid run -p quick -r 5  # this will produce a file like 20260731-120000.yuclid.jsonl

yuclid tplot 20260731-120000.yuclid.jsonl -x workers -z schedule -y tested_per_second \
  -f segment_kib=64KiB
yuclid tplot 20260731-120000.yuclid.jsonl -x segment_kib -z workers -y peak_rss_kib \
  -f schedule=dynamic
```

`-m sd,2` draws two standard deviations around each point instead of the
default percentile interval, and `-m none` removes the spread.

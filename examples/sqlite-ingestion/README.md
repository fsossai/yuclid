# SQLite ingestion

Imagine having to load a large CSV into SQLite. There are three ways to write
the inserts, three journal modes to combine them with, and batching to tune —
but batching only means something for one of the three ways.

This example shows how to carve those impossible combinations out of the space,
and how to give every point a private file to work in.

## What to look at in `yuclid.json`

**Conditions that depend on another dimension.** `batch` exists only when
`strategy` is `executemany`, and the largest row count is skipped for
`autocommit`. A condition is an expression over the point, so it can mention
any dimension: `yuclid.strategy != 'autocommit'`.

**A private file per point.** Each trial writes its database to
`${yuclid.@}.sqlite`. `${yuclid.@}` is a unique name for the point, the same
one used for the captured output, so two points can never collide and the
databases are still there afterwards.

**Combining dimensions when plotting.** `strategy` and `batch` only mean
something together. `-C strategy,batch` merges them into one axis called
`strategy_batch` at plotting time — the configuration does not have to
anticipate it.

**Timing the phases separately.** Insert, index and query are three metrics
from one trial, so you can ask which phase a setting actually changes.

## Running it

```sh
yuclid run --dry-run
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -C strategy,batch -x strategy_batch -z journal \
  -y rows_per_second
yuclid tplot yuclid.results.jsonl -C strategy,batch -x records -z strategy_batch \
  -y peak_rss_kib -f journal=WAL
```

`--show-missing` lists the combinations that are absent, which is a good way to
check that the conditions carved out what you meant.

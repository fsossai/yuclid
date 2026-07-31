# SQLite ingestion

Imagine having to load a large CSV into SQLite. There are three ways to write
the inserts, three journal modes to combine them with, and a batch size to
tune. Batching only applies to one of the three ways, and the slowest way is
not worth running on the largest input.

## What to look at in `yuclid.json`

**Conditions over other dimensions.** A condition is an expression over the
whole point, so it can mention any dimension. `batch` exists only when
`strategy` is `executemany`, and the largest row count is skipped when
`strategy` is `autocommit`.

**A private file per point.** Each trial writes its database to
`${yuclid.@}.sqlite`. `${yuclid.@}` is the identifier of the point, the same
one used for the captured output, so two points never write to the same file
and the databases remain afterwards.

**Combining dimensions when plotting.** `strategy` and `batch` describe one
choice between them. `-C strategy,batch` merges them into a single axis named
`strategy_batch` at plotting time, without changing the configuration.

**Phases as separate metrics.** Insert, index and query are timed separately by
three metrics reading the same trial output.

## Running it

```sh
yuclid run --dry-run
yuclid run -p quick  # this will produce yuclid.results.20260731-120000.jsonl

yuclid tplot yuclid.results.20260731-120000.jsonl -C strategy,batch -x strategy_batch -z journal \
  -y rows_per_second
yuclid tplot yuclid.results.20260731-120000.jsonl -C strategy,batch -x records -z strategy_batch \
  -y peak_rss_kib -f journal=WAL
```

`--show-missing` lists the combinations that are absent from the results, which
is one way to check what the conditions removed.

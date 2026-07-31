# SQLite ingestion strategies

The same deterministic event stream is loaded into SQLite three ways — one
autocommitted insert per row, individual inserts inside one transaction, and
batched `executemany` inside one transaction — across the `DELETE`, `WAL` and
`OFF` journal modes. Ingestion, index creation and an aggregate query are timed
separately.

Two conditions carve the space:

- `batch` is meaningful only to `executemany`, and is `none` otherwise;
- 100k rows are skipped for `autocommit`, where one commit per row would
  dominate the run without adding anything.

Each trial writes its database at `${yuclid.@}.sqlite`, next to that point's
own captures, so points never share a file and the databases survive the run.

Because `strategy` and `batch` only mean something together, combine them into
one axis when plotting:

```sh
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -C strategy,batch -x strategy_batch -z journal \
  -y rows_per_second
yuclid tplot yuclid.results.jsonl -C strategy,batch -x records -z strategy_batch \
  -y peak_rss_kib -f journal=WAL
```

The `scaling` preset goes to 10k and 100k rows over the two faster journal
modes.

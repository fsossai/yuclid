# Examples

Each directory is a self-contained experiment: a `yuclid.json` with an
equivalent `yuclid.yaml` beside it, the workload it measures, and no
third-party dependencies.

| Example | Goal |
|---|---|
| [matrix-multiplication](matrix-multiplication/) | How loop order and tiling change dense matrix multiplication |
| [cache-traversal](cache-traversal/) | What access pattern, stride and working-set size do to memory performance |
| [compression-codecs](compression-codecs/) | How gzip, bzip2 and LZMA trade speed, ratio and memory |
| [sqlite-ingestion](sqlite-ingestion/) | How transactions, batching and journaling affect SQLite loading |
| [parallel-prime-sieve](parallel-prime-sieve/) | How worker count, scheduling and segment size affect multicore scaling |

```sh
cd examples/<example>
yuclid run --dry-run              # every command that would run, run nothing
yuclid run -i yuclid.yaml -p quick   # the YAML twin, same experiment
yuclid run -p quick -o yuclid.results.jsonl
yuclid tplot yuclid.results.jsonl -x <dimension> -z <dimension> -y <metric>
```

Every example has a `quick` preset that finishes in seconds and a larger one
worth running when you care about the numbers. Add `-r 5` to repeat each point
five times.

Generated inputs and binaries land in each example's `data/` and `build/`.

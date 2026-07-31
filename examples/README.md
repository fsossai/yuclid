# Examples

Five experiments, each in its own directory, written to teach one part of
yuclid at a time. Every directory holds a `yuclid.json` with an equivalent
`yuclid.yaml` beside it, the workload it measures, and no third-party
dependencies.

| Example | Goal |
|---|---|
| [matrix-multiplication](matrix-multiplication/) | Teach a plain space, point setup on one dimension, and presets |
| [cache-traversal](cache-traversal/) | Teach conditions, several metrics from one trial, and wrapping a workload |
| [compression-codecs](compression-codecs/) | Teach point setup shared across a subspace, and metrics that are not timings |
| [sqlite-ingestion](sqlite-ingestion/) | Teach conditions over other dimensions, and a private file per point |
| [parallel-prime-sieve](parallel-prime-sieve/) | Teach repetitions, and what to do about a noisy measurement |

Read them in that order if you are new to yuclid.

## The usual loop

```sh
cd examples/<example>
yuclid run --dry-run                  # every command that would run, run nothing
yuclid run -p quick -o yuclid.results.jsonl
yuclid tplot yuclid.results.jsonl -x <dimension> -z <dimension> -y <metric>
```

`--dry-run` resolves the whole configuration and prints the commands without
executing any of them. Every example defines a `quick` preset covering a small
part of its space; dropping `-p quick` runs the whole space.

`-i yuclid.yaml` runs the YAML twin instead, `-r 5` repeats each point five
times, and `--compile run.sh` writes the experiment out as a plain shell
script.

Generated inputs and binaries land in each example's `data/` and `build/`.

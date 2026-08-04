# Examples

Three experiments, each in its own directory, written to teach one part of
yuclid at a time. Every directory holds a `yuclid.json` with an equivalent
`yuclid.yaml` beside it, and the workload it measures.

| Example | Goal |
|---|---|
| [matrix-multiplication](matrix-multiplication/) | Teach a plain space, point setup on one dimension, and presets |
| [compression-codecs](compression-codecs/) | Teach point setup shared across a subspace, a dimension the user supplies, and metrics that are not timings |
| [pointer-structures](pointer-structures/) | Teach one program run two ways, derived metrics, and hardware counters |

Read them in that order if you are new to yuclid.

The first two need only a C compiler and Python. `pointer-structures` measures
hardware counters, so it needs Linux with `perf` and `strace`.

## The usual loop

```sh
cd examples/<example>
yuclid run --dry-run  # every command that would run, run nothing
yuclid run -p quick   # this will produce a file like 20260731-120000.yuclid.jsonl
yuclid describe 20260731-120000.yuclid.jsonl
yuclid tplot 20260731-120000.yuclid.jsonl -x <dimension> -z <dimension> -y <metric>
```

`--dry-run` resolves the whole configuration and prints the commands without
executing any of them. Every example defines a `quick` preset covering a small
part of its space; dropping `-p quick` runs the whole space.

The name of the file is printed at the end of the run. Pass `-o` to choose it
yourself.

`-i yuclid.yaml` runs the YAML twin instead, `-r 5` repeats each point five
times, and `--compile run.sh` writes the experiment out as a plain shell
script.

Generated inputs and binaries land in each example's `data/` and `build/`.

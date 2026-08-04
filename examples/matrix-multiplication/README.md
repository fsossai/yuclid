# Matrix multiplication

Imagine having to find out which way of writing a matrix multiplication is
fastest. There are three loop orders to try, each of them with and without
cache blocking, on matrices of several sizes. That is one program run per
combination, and a number to collect from each.

## The space

The space has three dimensions: `size`, `variant` and `tile`. Every
combination of the three is a point, and yuclid runs the trial once per point.

A matrix file has to exist before the trials can use it, and
it depends on `size` alone.
The `on: ["size"]` list says so, and the file is generated once per size rather than once per point. `parallel: true` allows
those generations to run at the same time.
The `quick` preset can be run with `yuclid run -p quick`.
`order` lists `size` first, so yuclid varies it slowest and finishes
everything about one matrix size before moving to the next.

Set `"size": null` to require the user to provide a number via `-s`, for example `yuclid run -s size=100,200`.



```sh
yuclid run
yuclid run --dry-run
yuclid run --preset quick
yuclid run --select variant=dot size=512  # cut other variants and sizes
yuclid run --select tile=32x32
yuclid run --repeat 3                     # 3 runs per point

# at this point, a file like 20260731-120000.yuclid.jsonl
# will be available

# simple visualization, the y axis is inferred (e.g., seconds)
yuclid tplot 20260731-120000.yuclid.jsonl -x variant -z tile

# normalize each `variant` by its version without tiling
yuclid tplot 20260731-120000.yuclid.jsonl -x variant -z tile -X tile=none

# find which configuration, overall, has the best speedup w.r.t. the dot variant with no tiling
yuclid plot 20260731-120000.yuclid.jsonl -x variant -z tile -R variant=dot tile=none -r -A

```

The arrow keys move through the dimensions that are not on the plot.

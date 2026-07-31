# Matrix multiplication

Imagine having to find out which way of writing a matrix multiplication is
fastest. There are three loop orders to try, each of them with and without
cache blocking, on matrices of several sizes. That is one program run per
combination, and a number to collect from each.

## What to look at in `yuclid.json`

**The space.** Three dimensions: `size`, `variant` and `tile`. Every
combination of the three is a point, and yuclid runs the trial once per point.

**Names against values.** `size` is written as `name`/`value` pairs. The
command receives the value `512`, and the results record the name `512x512`.
The same applies to `tile`, whose value `0` is named `none`.

**Point setup.** The matrix file has to exist before the trials can use it, and
it depends on `size` alone. The `on: ["size"]` list says so, and the file is
generated once per size rather than once per point. `parallel: true` allows
those generations to run at the same time.

**Presets.** `quick` restricts `size` to the smallest one; `large` selects the
other two. A preset is a named subspace, and `-p` runs it.

**Order.** `order` lists `size` first, so yuclid varies it slowest and finishes
everything about one matrix size before moving to the next.

Needs a C11 compiler as `cc`.

## Running it

```sh
yuclid run --dry-run  # print every command, run none of them
yuclid run -p quick   # this will produce a file like yuclid.results.20260731-120000.jsonl
yuclid run

yuclid tplot yuclid.results.20260731-120000.jsonl -x variant -z tile -y seconds
yuclid tplot yuclid.results.20260731-120000.jsonl -x size -z variant -y seconds -f tile=none

yuclid plot yuclid.results.20260731-120000.jsonl \
  -x variant -z tile -R variant=dot tile=none -r -A
```

The arrow keys move through the dimensions that are not on the plot.

The last command divides every value by the one at `variant=dot tile=none`,
so the plot reads as a speedup against the plain dot product. `-R` names the
reference point, `-r` turns the ratio the right way up, and `-A` writes the
number on each bar.

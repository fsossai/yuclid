# Matrix multiplication

Imagine having to find out which way of writing a matrix multiplication is
fastest. There are three loop orders to try, each of them with and without
cache blocking, on matrices of several sizes. That is one program run per
combination, and a number to collect from each.

This is what yuclid is for. The configuration declares the three things that
vary, and yuclid runs the rest.

## What to look at in `yuclid.json`

**The space.** Three dimensions: `size`, `variant` and `tile`. Every
combination of the three is a point, so the whole grid comes from nine lines
of configuration.

**Names against values.** `size` is written as `name`/`value` pairs: the
command receives `512`, while the results say `512x512`. The same for `tile`,
where the value `0` is called `none`.

**Point setup.** Each matrix file has to exist before the trials can use it,
but it depends on `size` alone. The `on: ["size"]` list says so, and the file
is generated once per size instead of once per run. `parallel: true` lets those
generations happen at the same time.

**Presets.** `quick` restricts `size` to the smallest one. Use it while you are
still writing the configuration, then run the whole space when you mean it.

**Order.** `order` puts `size` first, so yuclid finishes everything about one
matrix size before moving to the next.

Needs a C11 compiler as `cc`.

## Running it

```sh
yuclid run --dry-run              # print every command, run none of them
yuclid run -p quick -o yuclid.results.jsonl
yuclid run -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x variant -z tile -y seconds
yuclid tplot yuclid.results.jsonl -x size -z variant -y seconds -f tile=none
```

Move through the slices with the arrow keys.

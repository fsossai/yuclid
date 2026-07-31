# Matrix multiplication loop order

Three mathematically equivalent dense `C = A × B` kernels, each measured with
and without cache blocking, over three matrix sizes:

- `dot` — one dot product per output element (`i-j-k`);
- `rows` — linear combinations of rows (`i-k-j`);
- `columns` — linear combinations of columns (`j-k-i`).

`tile` is a parameter of every kernel rather than a kernel of its own: `none`
runs the plain loop nest, and a tile size runs the same loop order applied
first to blocks and then within a block. The space is the full product —
3 sizes × 3 loop orders × 5 tile settings.

The two questions it separates are worth keeping apart. **Loop order** decides
the order of magnitude: at 1024×1024, `columns` takes about 9.6 s against
0.23 s for `rows`, a factor of 40 between nests that compute exactly the same
thing. **Blocking** then buys back a factor of roughly two on the orders that
traverse memory badly — `dot` goes from 1.84 s to 0.96 s and `columns` from
9.6 s to 5.8 s at 32×32 tiles — while costing `rows` about 20%, because a
kernel already walking memory sequentially gains nothing and pays the extra
loop overhead.

The matrices are large on purpose: at 512×512 the fastest kernel already takes
tens of milliseconds, so what is measured is the kernel rather than the timer's
noise floor. Each matrix pair is generated once per size from a fixed seed, so
repeated runs compare against byte-identical inputs.

Needs a C11 compiler as `cc`.

```sh
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x variant -z tile -y seconds
yuclid tplot yuclid.results.jsonl -x size -z variant -y seconds -f tile=none
```

`quick` is 512×512 alone: 15 points in about 5 seconds. `large` adds the two
bigger sizes and takes a few minutes, most of it spent in untiled `columns`.

The generated matrices occupy about 57 MB in `data/` once every size has been
visited.

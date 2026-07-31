# Matrix multiplication loop order

Four mathematically equivalent dense `C = A × B` kernels, over three matrix
sizes from 512×512 to 1536×1536:

- `dot` — one dot product per output element (`i-j-k`);
- `rows` — linear combinations of rows (`i-k-j`);
- `columns` — linear combinations of columns (`j-k-i`);
- `tiled` — cache-blocked `i-k-j`, at four tile sizes.

The matrices are large on purpose: at 512×512 the fastest kernel already takes
tens of milliseconds, so what is measured is the kernel rather than the timer's
noise floor. What comes out is dominated by the loop order — at 1024×1024,
`columns` takes about 9 s against 0.23 s for `rows`, a factor of 40 between two
nests that compute exactly the same thing.

The tile size is a second-order effect by comparison: at 1024×1024 it buys
about 25% going from 32×32 to 256×256 tiles, and by 1536×1536 the four tile
sizes are within a few percent of each other.

`tile` exists only for `tiled`; the other variants carry the single value
`none`. The space is therefore 3 sizes × 7 variant/tile combinations, not the
full product of 3 × 4 × 5.

Each matrix pair is generated once per size from a fixed seed, so repeated runs
compare against byte-identical inputs.

Needs a C11 compiler as `cc`.

```sh
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x size -z variant -y seconds -f tile=none
yuclid tplot yuclid.results.jsonl -x size -z tile -y seconds -f variant=tiled
```

`quick` is 512×512 alone and takes a couple of seconds. `large` adds the two
bigger sizes and takes about a minute, most of it spent in `columns`. Running
every size takes roughly the same and produces the full scaling curve.

The generated matrices occupy about 57 MB in `data/` once every size has been
visited.

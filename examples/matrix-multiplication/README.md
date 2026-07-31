# Matrix multiplication loop order

Four mathematically equivalent dense `C = A × B` kernels, over four matrix
sizes from 512×512 to 2048×2048:

- `dot` — one dot product per output element (`i-j-k`);
- `rows` — linear combinations of rows (`i-k-j`);
- `columns` — linear combinations of columns (`j-k-i`);
- `tiled` — cache-blocked `i-k-j`, at four tile sizes.

The matrices are large on purpose. At 512×512 the fastest kernel already takes
tens of milliseconds and the slowest takes a second, so the differences are the
measurement rather than the timer's noise floor. They are dramatic: at
1024×1024 `columns` takes about 9 s against 0.23 s for `rows`, a factor of 40
between two loop nests that compute exactly the same thing.

`tile` exists only for `tiled`; the other variants carry the single value
`none`. The space is therefore 4 sizes × 7 variant/tile combinations, not the
full product of 4 × 4 × 5.

`checksum` is collected next to `seconds`, and all four variants must report
the same value — a kernel that is fast because it is wrong shows up in the
results rather than in the plot. Each matrix pair is generated once per size
from a fixed seed, so repeated runs compare against byte-identical inputs.

Needs a C11 compiler as `cc`.

```sh
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x size -z variant -y seconds -f tile=none
yuclid tplot yuclid.results.jsonl -x size -z tile -y seconds -f variant=tiled
```

`quick` covers 512 and 1024 in about 20 seconds. `large` adds 1536 and 2048 and
takes a few minutes, most of it spent in `columns`; that is where tiling starts
to pay, dropping from 3.9 s at 32×32 tiles to 2.3 s at 256×256.

The generated matrices occupy roughly 120 MB in `data/` once every size has
been visited.

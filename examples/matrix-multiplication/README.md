# Matrix multiplication loop order

Four mathematically equivalent dense `C = A × B` kernels, over four matrix
sizes:

- `dot` — one dot product per output element (`i-j-k`);
- `rows` — linear combinations of rows (`i-k-j`);
- `columns` — linear combinations of columns (`j-k-i`);
- `tiled` — cache-blocked `i-k-j`, at three tile sizes.

`tile` exists only for `tiled`; the other variants carry the single value
`none`. The space is therefore 4 sizes × 6 variant/tile combinations, not the
full product of 4 × 4 × 4.

`checksum` is collected next to `seconds`, and all four variants must report
the same value — a kernel that is fast because it is wrong shows up in the
results rather than in the plot. Each matrix pair is generated once per size
from a fixed seed, so repeated runs compare against byte-identical inputs.

Needs a C11 compiler as `cc`.

```sh
yuclid run -p quick -o results.jsonl

yuclid tplot results.jsonl -x size -z variant -y seconds -f tile=none
yuclid tplot results.jsonl -x size -z tile -y seconds -f variant=tiled
```

The `large` preset reaches 512×512, where the loop orders separate clearly.

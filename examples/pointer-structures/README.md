# Pointer structures, under the counters

Imagine having to answer why one container is faster than another.
This example measures a sorted linked list,
a binary search tree and a hash table with chaining doing the same five things,
each built at two optimisation levels,
and collects six numbers about every one.
The metrics are operations completed, instructions per cycle, cache hit rate,
branch predictability, nodes visited, and how often the program went to the
kernel for memory.

## The workload

`structures.c` builds the structure once and then repeats the chosen operation
until `$BUDGET_MS` has elapsed, reporting how many it managed.

The operations are `build`, `lookup`, `absent` (searching for keys that are not
there — a different branch profile), `traverse`, and `churn` (inserting a key
and taking it straight out again, so the container ends each operation the size
it started).


## The space

Three structures, five operations, three sizes, two optimisation levels: the
full product, 90 points.

The dimension `opt` is `O2` or `O3`, and each needs a compiler
invocation of its own in `setup.point`.

```json
"point": [
  { "command": "mkdir -p build/${yuclid.opt}", "on": ["opt"], "parallel": true },
  { "command": "cc $CFLAGS -${yuclid.opt} structures.c -o build/${yuclid.opt}/structures",
    "on": ["opt"], "parallel": true }
]
```

Both stages are `parallel`, since the two levels write to different paths and cannot collide.

The metric `visits_per_op` carries `"condition": "yuclid.operation in ['lookup', 'absent']"`. Only a search has an
interesting number of nodes to visit — a traversal touches all of them by
definition — so the column exists only where it means something. The trial
still runs for the other operations, because its other metrics are still valid
there.

## Two ways of running one program

The metric `alloc_syscalls` counts the times the allocator asked
the kernel for memory, and it cannot come from the same invocation with `perf` like the others.
So there are two trials over the same program, each declaring the metrics it feeds:

```sh
yuclid run                      # both: five columns from perf, one from strace
yuclid run -m ops ipc           # only the perf trial runs
yuclid run -m alloc_syscalls    # only strace runs, and it is the whole run
```


## Running it

```sh
yuclid run --dry-run   # 90 points, and two compilations
yuclid run -p quick    # this will produce a file like 20260731-120000.yuclid.jsonl

yuclid describe 20260731-120000.yuclid.jsonl
```

The comparisons the example exists to make:

```sh
# the same search, and how many of it fit in the budget
yuclid plot 20260731-120000.yuclid.jsonl -x nodes -z structure -y ops -f operation=lookup opt=O3

# an algorithmic count beside a hardware one: visits explain some of it, not all
yuclid plot 20260731-120000.yuclid.jsonl -x nodes -z structure -y visits_per_op
yuclid plot 20260731-120000.yuclid.jsonl -x nodes -z structure -y cache_hit_rate

# what the optimiser was worth, which is not the same for every container
yuclid plot 20260731-120000.yuclid.jsonl -x structure -z opt -y ipc -f operation=traverse nodes=1M
```

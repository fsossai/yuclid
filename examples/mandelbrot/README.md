# Parallel Mandelbrot

This example measures strong scaling and load balance while rendering a fixed
Mandelbrot image. It compares GCC and Clang across several thread counts and
OpenMP schedules.

## The Space

The space has three dimensions: `compiler`, `threads`, and `schedule`. The
non-static schedules only apply when more than one thread is used, so those
combinations are excluded from the space at one thread.

Each compiler is built once in `setup.point`. The trials then record elapsed
time, throughput in millions of iterations per second, the number of threads
used, and load imbalance. An imbalance of `1.0` means that the work was divided
evenly; larger values mean that the busiest thread ran longer than the average
thread.

The `quick` preset samples two thread counts and two schedules. The `scaling`
preset keeps the static and guided schedules across every thread count.

```sh
yuclid run
yuclid run --dry-run
yuclid run --preset quick
yuclid run --preset scaling
yuclid run --select compiler=gcc,clang schedule=static
yuclid run --select threads=4,8
yuclid run --repeat 3

# At this point, a file like 20260731-120000.yuclid.jsonl is available.

# Compare schedules as the number of threads grows.
yuclid tplot 20260731-120000.yuclid.jsonl -x threads -z schedule -y seconds

# Show how much each schedule gains over static scheduling at each thread count.
yuclid plot 20260731-120000.yuclid.jsonl -x threads -z schedule -y seconds -X schedule=static -r -A

# Show strong scaling for each compiler using the static schedule.
yuclid plot 20260731-120000.yuclid.jsonl -x threads -z compiler -y seconds -L schedule=static -Z threads=1 -r -A

# See whether slower runs correspond to uneven work distribution.
yuclid tplot 20260731-120000.yuclid.jsonl -x threads -z schedule -y imbalance
```

The arrow keys move through dimensions that are not on the plot. Run
`yuclid serve` in another terminal to monitor the space while the experiment
is running.

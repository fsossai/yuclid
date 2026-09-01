# Examples

Each directory contains a `yuclid.json` and the workload it measures.

- [Matrix multiplication](matrix-multiplication/) — compares loop orders and
  cache tiling across matrix sizes.
- [Compression codecs](compression-codecs/) — compares compression time and
  output size across codecs and levels.
- [Parallel Mandelbrot](mandelbrot/) — measures strong scaling and load balance
  across compilers and OpenMP schedules.
- [Pointer structures](pointer-structures/) — compares traversal strategies
  using timings and hardware counters.

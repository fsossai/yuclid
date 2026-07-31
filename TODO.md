# TODO

- [ ] feat: adaptive repetition (repeat until CI on the median is within a target)
- [ ] feat: metric fallback value when extraction finds nothing, instead of erroring
- [ ] feat: Parquet export / `yuclid export`
- [ ] feat: provenance in output — yuclid version, config hash, host, timestamp, per-point exit code and wall time
- [ ] feat: publish a JSON Schema (free editor autocomplete + validation)
- [ ] feat: sum and subtract presets
- [ ] feat: web viewer
- [ ] fix: conditions can alter the expected order of the experiments
- [ ] fix: evaluation order of variables in `env`
- [ ] fix: compiled script should print status
- [ ] new: `--fold` pads the shorter metrics of a point with `NaN`
- [ ] new: a metric enabled by several trials should be reported
- [ ] new: add examples for real benchmark suites or programs (e.g., GAPBS)
- [ ] new: turn `panorama` into a skill

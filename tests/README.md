# Tests for `yuclid run`

```
python tests/run_tests.py            # run everything (~20s)
python tests/run_tests.py --list     # what each case covers
python tests/run_tests.py -k preset  # only cases whose name contains "preset"
python tests/run_tests.py -k global-setup -v --keep   # show yuclid's output, keep the workdir
python tests/run_tests.py -j 1       # one case at a time
```

Cases share nothing, so they run concurrently (`-j`, default: one per core).
The wall time is dominated by process start-up: a `yuclid` invocation spends
about 1.5s importing seaborn and scipy through `yuclid.spread` before `run`
does anything, and none of that is needed by `run`. A single invocation may
take `--timeout` seconds (60 by default) before the case is failed, so a hang
cannot block the suite.

No dependencies beyond the standard library and whatever `yuclid` itself needs.
The runner picks an interpreter that can import the **working copy** of yuclid
(`PYTHONPATH` points at the repository, so nothing installed is used); override
it with `--python` or `YUCLID_TEST_PYTHON`.

## How a case works

One directory per case under `cases/`:

```
cases/global-setup-runs-once/
    yuclid.json     the configuration under test
    case.json       how to invoke yuclid and what must come out
```

Every file except `case.json` is copied into a fresh temporary directory, and
yuclid runs there. A case therefore never sees the repository, the user's files
or another case's leftovers.

**Predictability.** Trials are deterministic shell one-liners — `echo`,
`printf`, `wc`, `awk` — chosen so that every metric has a value fixed by the
configuration alone. Nothing is timed, nothing is random, nothing reaches the
network, and no case depends on the host's CPU count, locale or clock. The
expected records are therefore written out literally.

## `case.json`

| Key | Meaning |
|---|---|
| `description` | what the case pins down (shown by `--list`) |
| `args` | argv after `yuclid` (default `["run"]`) |
| `runs` | several invocations in sequence, instead of `args` |
| `output_flag` | set `false` to stop the runner appending `-o results.jsonl` |
| `exit_code` / `exit_codes` | expected status of the last run / of each run |
| `records` | the exact dataset, in order |
| `records_unordered` | the exact dataset, as a set |
| `record_count` | how many records |
| `no_results_file` | no dataset may be written |
| `stdout_contains` / `stdout_not_contains` | substrings of the diagnostics |
| `files_exist` / `files_not_exist` | globs, relative to the working directory |
| `xfail` | why this case is expected to fail today |

Numbers are compared strictly: an expected `3` does not match an actual `3.0`.
A `NaN` in the dataset is compared as the string `"NaN"`.

## Known defects

Cases carrying `xfail` describe behaviour the tool is *supposed* to have and
does not; the runner reports them as `XFAIL` and stays green. If one starts
passing, it is reported as `XPASS` and the suite fails — delete the `xfail` key
and the case becomes a regression test. Each `xfail` string says what actually
goes wrong.

Currently expected to fail:

| Case | Defect |
|---|---|
| `preset-wildcard` | a `*` preset entry collects names instead of values and crashes |
| `numeric-types` | one float metric widens every other metric of the same point |
| `point-setup-as-string` | `setup.point` as a plain string crashes |
| `per-value-setup-leaks-across-presets` | a value's setup re-runs in later presets |
| `preset-defines-undefined-dimension` | a preset cannot define an undefined dimension |
| `point-setup-condition-outside-on` | conditions referring outside `on` raise instead of being diagnosed |

## Note on invocation

The runner starts yuclid exactly as the installed console script does
(`from yuclid.cli import main`). `python -m yuclid.cli` is **not** equivalent —
it hits a circular import between `yuclid.cli`, `yuclid.plot` and
`yuclid.tplot`.

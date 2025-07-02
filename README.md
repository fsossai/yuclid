# Yuclid

*Combinatorially explode your experiments*

## Installation

Current status:
```
pip install git+https://github.com/fsossai/yuclid.git
```

Latest release:
```
pip install yuclid
```

- **`yuclid run`**: Run experiments with with all combination of registered parameters.
- **`yuclid plot`**: Interactively visualizes the results produced by `yuclid run`.

## Configuration for `yuclid run`

Key sections:
- **`env`**: Environment variables and constants
- **`setup`**: Commands to run before experiments (`global`) or for specific parameter combinations (`point`)
- **`trials`**: The actual experiment commands that generate metrics to collect
- **`metrics`**: How to extract a give metric from the data collected by the trials
- **`space`**: Dimension definitions - all combinations will be explored
- **`order`**: Execution order of parameter combinations

Parameters can be simple lists or objects with `name`/`value` pairs.
Use `${yuclid.x}` to reference dimension values in commands, and `${yuclid.@}` for unique output filenames.
`${yuclid.x}` is an alias for `${yuclid.x.value}`.

## Minimal Example

Yuclid uses a `yuclid.json` configuration file to define experiment parameters and execution settings.
Here's a minimal example that you can immediately run on your linux terminal.

```json
{
  "space": {
    "size": [
      {
        "name": "small",
        "value": 100000
      },
      {
        "name": "medium",
        "value": 1000000
      },
      {
        "name": "large",
        "value": 10000000
      }
    ],
    "hash": [
      "md5sum",
      "sha256sum"
    ],
    "cpuid": [0, 1, 2, 3]
  },
  "trials": [
    "time -p taskset -c ${yuclid.cpuid} head -${yuclid.size} /dev/urandom | ${yuclid.hash}"
  ],
  "metrics": [
    {
      "name": "time.real",
      "command": "cat ${yuclid.@}.err | grep real | grep -oE '[0-9]+\\.[0-9]+'"
    },
    {
      "name": "time.sys",
      "command": "cat ${yuclid.@}.err | grep sys | grep -oE '[0-9]+\\.[0-9]+'"
    }
  ]
}
```
The command `yuclid run` (or `yuclid run --inputs yuclid.json`) will produce a JSON Lines:
```json
{"size": "medium", "hash": "md5sum", "cpuid": "0", "time.real": 1.35, "time.sys": 1.19}
{"size": "medium", "hash": "md5sum", "cpuid": "1", "time.real": 1.36, "time.sys": 1.2}
{"size": "medium", "hash": "md5sum", "cpuid": "2", "time.real": 1.35, "time.sys": 1.2}
{"size": "medium", "hash": "md5sum", "cpuid": "3", "time.real": 1.33, "time.sys": 1.18}
{"size": "medium", "hash": "sha256sum", "cpuid": "0", "time.real": 1.37, "time.sys": 1.19}
{"size": "medium", "hash": "sha256sum", "cpuid": "1", "time.real": 1.36, "time.sys": 1.19}
{"size": "medium", "hash": "sha256sum", "cpuid": "2", "time.real": 1.39, "time.sys": 1.2}
{"size": "medium", "hash": "sha256sum", "cpuid": "3", "time.real": 1.37, "time.sys": 1.19}
{"size": "small", "hash": "md5sum", "cpuid": "0", "time.real": 0.13, "time.sys": 0.12}
{"size": "small", "hash": "md5sum", "cpuid": "1", "time.real": 0.13, "time.sys": 0.12}
{"size": "small", "hash": "md5sum", "cpuid": "2", "time.real": 0.13, "time.sys": 0.12}
{"size": "small", "hash": "md5sum", "cpuid": "3", "time.real": 0.13, "time.sys": 0.12}
{"size": "small", "hash": "sha256sum", "cpuid": "0", "time.real": 0.14, "time.sys": 0.12}
{"size": "small", "hash": "sha256sum", "cpuid": "1", "time.real": 0.14, "time.sys": 0.12}
{"size": "small", "hash": "sha256sum", "cpuid": "2", "time.real": 0.14, "time.sys": 0.12}
{"size": "small", "hash": "sha256sum", "cpuid": "3", "time.real": 0.14, "time.sys": 0.12}
{"size": "large", "hash": "md5sum", "cpuid": "0", "time.real": 13.29, "time.sys": 11.74}
{"size": "large", "hash": "md5sum", "cpuid": "1", "time.real": 13.34, "time.sys": 11.82}
{"size": "large", "hash": "md5sum", "cpuid": "2", "time.real": 13.38, "time.sys": 11.81}
{"size": "large", "hash": "md5sum", "cpuid": "3", "time.real": 13.31, "time.sys": 11.74}
{"size": "large", "hash": "sha256sum", "cpuid": "0", "time.real": 13.61, "time.sys": 11.81}
{"size": "large", "hash": "sha256sum", "cpuid": "1", "time.real": 13.66, "time.sys": 11.88}
{"size": "large", "hash": "sha256sum", "cpuid": "2", "time.real": 13.84, "time.sys": 12.02}
{"size": "large", "hash": "sha256sum", "cpuid": "3", "time.real": 13.62, "time.sys": 11.82}

```
These results can be displayed with `yuclid plot`, e.g.:
```
yuclid plot results.json -x size
yuclid plot results.json -x hash
yuclid plot results.json -x size -z cpuid
```
The same configuration can be executed on slices of the space (i.e. subspaces):
```
yuclid run -s size=medium
yuclid run -s cpuid=0,1,2
yuclid run -s size=small,medium cpuid=3,0
```
Check out `yuclid run -h` for more info.

## Advanced Example


```json
{
  "env": {
    "root": "/my/path",
    "data_dir": "/path/to/data"
  },
  "setup": {
    "global": [
      "ulimit -s 1048576" // global commands are run before point commands
    ],
    "point": [
      {
        "on": [ "compiler" ], // run the command on these dimensions only.
                              // The entire space is assumed if empty.
        "command": "mkdir -p ${yuclid.compiler}",
        "parallel": [ "compiler" ] // list|true|false: can execute more commands in parallel
                                   // true = all dimensions in `on`.
      },
      {
        "on": [ "compiler" ], // run the command on these dimensions only.
                              // The entire space is assumed if empty.
        "command": "make myprogram.out CXX=${yuclid.compiler} OUTDIR=$root/build/${yuclid.compiler}",
        "parallel": true // equivalent to ["compiler"]
      }
    ]
  },
  "space": {
    "compiler": [ "g++", "clang++" ],
    "threads": [ 1, 2, 3, 4 ],
    // or
    "threads:py": "list(range(1,5))", // python!
    // or
    "nthreads": null, // this forces the user to specify nthreads from CLI
                      // e.g. --select nthreads=1,7,14
    "dataset": [
      {
        "name": "small",
        "value": "${data_dir}/mydatasetA.dat",
        "condition": "yuclid.nthreads == 1"
      },
      {
        "name": "small", // name can be duplicated
        "value": "${data_dir}/mydatasetB.dat",
        "condition": "yuclid.nthreads > 1"
      }
    ]
  },
  "trials": [
    {
      "command": "time -p ${yuclid.compiler}/myprogram.out ${yuclid.dataset}",
      "metrics": [ "time", "something_else" ] // which metrics this command enables
                                              // "condition": "True" can specify extra conditions
    }
  ],
  "metrics": [
    {
      "name": "time",
      // each metric command must generate one or more numbers (separated by space or linebreak)
      // ${yuclid.@} represents a unique trial identifier
      // ${yuclid.@}.out and ${yuclid.@}.err are automatically generated for each trial
      "command": "cat ${yuclid.@}.err | grep real | grep -E '[0-9]+\\.[0-9]+'"
    },
    {
      "name": "something_else",
      "command": "cat ${yuclid.@}.out | grep something"
    }
  ],
  "order": [ "compiler", "dataset", "nthreads" ]  // different nthreads first,
                                                  // then datasets, then compilers
}
```


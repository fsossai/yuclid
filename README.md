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
    ]
  },
  "trials": [
    "{ time -p cat /dev/urandom | head -${yuclid.size} | md5sum ; } 2>&1"
  ],
  "metrics": [
    {
      "name": "time.real",
      "command": "cat ${yuclid.@}.out | grep real | grep -oE '[0-9]+\\.[0-9]+'"
    },
    {
      "name": "time.sys",
      "command": "cat ${yuclid.@}.out | grep sys | grep -oE '[0-9]+\\.[0-9]+'"
    }
  ]
}
```


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
        "on": ["compiler"], // run the command on these dimensions only.
                            // The entire space is assumed if empty.
        "command": "mkdir -p ${yuclid.compiler}",
        "parallel": ["compiler"] // list|true|false: can execute more commands in parallel
                                 // true = all dimensions in `on`.
      },
      {
        "on": ["compiler"], // run the command on these dimensions only.
                            // The entire space is assumed if empty.
        "command": "make myprogram.out CXX=${yuclid.compiler} OUTDIR=$root/build/${yuclid.compiler}",
        "parallel": true // equivalent to ["compiler"]
      },
    ]
  },
  "trials": [
    {
      "command": "{ time -p ${yuclid.compiler}/myprogram.out ${yuclid.dataset} ; } 2>&1",
      "metrics": ["time", "something_else"] // which metrics this command enables
      // "condition": "True" can specify extra conditions
    }
  ],
  "metrics": [
    {
        "name": "time",
        // each metric command must generate 1 or more lines contains
        // an interger or a floating point number.
        // ${yuclid.@} represents a unique trial identifier
        // ${yuclid.@}.out and ${yuclid.@}.err are automatically generated for each trial
        "command": "cat ${yuclid.@}.out | grep real | grep -E '[0-9]+\\.[0-9]+'"
    },
    {
        "name": "something_else",
        "command": "cat ${yuclid.@}.out | grep something"
    }
  ],
  "space": {
    "compiler": ["g++", "clang++"],
    "threads": [1, 2, 3, 4],
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
  "order": ["compiler", "dataset", "nthreads"] // different nthreads first,
                                               // then datasets, then compilers
}
```


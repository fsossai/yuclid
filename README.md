# Yuclid

*Combinatorially explode your experiments*

<p><img src="space.png" align="right" width="350" height="298"/></p>

Yuclid is a tool for orchestrating experiments in N-dimensional irregular spaces of parameters.
It collects custom metrics in a single JSON file for easy post-processing.
Yuclid builds the Cartesian product of the dimensions you defined, and runs an experiment per point in that space.
It also provides a unique way of plotting data (`yuclid plot`) interactively, browsing slices of the results using the arrow keys.

The **geometrical metaphor** is that each experiment is a point in a multidimensional discrete space formed by all combinations of user-defined parameters.
By specifying extra conditions (see advanced example), some hyper regions can be carved out of the original space.

## What kind of experiments?

Anything that can be expressed in a single (pipelined) command that generates one or more numbers.
Since programs' outputs are often verbose and the target metric is contained in a single line,
metrics can be arbitrarily defined in terms of other commands, e.g., regular expressions (see example).

Here's a list of use-case ideas:
- Measure the impact of different optimization levels of different **compilers** on different programs
- Count cache misses under different **memory allocators** on different inputs
- Measure strong scaling **parallel programs** given different thread affinities
- Evaluate different compression algorithms on different inputs with different compression levels
- Organize **perf** counters alongside custom metrics e.g., max RSS, in a self-contained JSON file
- Create reproducible artifacts for **research** software
- All of the above combined!

## Installation

Requires python >= 3.8

Development head:
```
pip install git+https://github.com/fsossai/yuclid.git
```

Stable release:
```
pip install yuclid
```

- **`yuclid run`**: Run experiments with all combinations of the defined parameters.
- **`yuclid plot`**: Interactively visualizes the results produced by `yuclid run`.

## Configuration for `yuclid run`

Key sections of `yuclid.json`:
- **`env`**: Environment variables and constants
- **`setup`**: Commands to run before experiments (`global`) or for specific parameter combinations (`point`)
- **`trials`**: The actual experiment commands that generate metrics to collect
- **`metrics`**: How to extract a given metric from the data collected by the trials
- **`space`**: Dimension definitions - all combinations will be explored
- **`order`**: Execution order of parameter combinations

Parameters can be simple lists or objects with `name`/`value` pairs.
Use `${yuclid.x}` in a command to reference the value of dimension `x`, and `${yuclid.@}` for a unique output filename.
`${yuclid.x}` is an alias for `${yuclid.x.value}`.

## Minimal Example

Suppose you want to time a compression algorithm on different input sizes and also measure the execution time variance across cores.
The dimensions of this experiment, i.e., the space, would be the _size_ of the input, the _compression_ level and the _cpuid_.
Yuclid uses a `yuclid.json` configuration file to define the space and other experiment parameters.
Here's a minimal example that you can immediately run on your linux terminal.

```json
{
  "space": {
    "size": [
      {
        "name": "small",
        "value": "10M"
      },
      {
        "name": "medium",
        "value": "20M"
      },
      {
        "name": "large",
        "value": "50M"
      }
    ],
    "cpuid": [0, 1, 2, 3],
    "compression": [
      {
        "name": "lowest",
        "value": 1
      },
      {
        "name": "highest",
        "value": 9
      }
    ]
  },
  "trials": [
    "time -p taskset -c ${yuclid.cpuid} head -c ${yuclid.size} /dev/urandom | gzip -${yuclid.compression} >/dev/null"
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
To run the experiments, copy the configuration above into `yuclid.json` and from the same directory run
```
yuclid run
```
You can also run a subspace using the selector `-s`
```
yuclid run -s size=medium
yuclid run -s cpuid=0,1,2
yuclid run -s size=small,medium cpuid=3,0
```

The command `yuclid run` (or `yuclid run --inputs yuclid.json`) will produce a JSON Lines:

```json
{"size": "small", "cpuid": "0", "compression": "lowest", "time.real": 0.37, "time.sys": 0.05}
{"size": "small", "cpuid": "0", "compression": "highest", "time.real": 0.33, "time.sys": 0.05}
{"size": "small", "cpuid": "1", "compression": "lowest", "time.real": 0.31, "time.sys": 0.05}
{"size": "small", "cpuid": "1", "compression": "highest", "time.real": 0.33, "time.sys": 0.05}
{"size": "small", "cpuid": "2", "compression": "lowest", "time.real": 0.31, "time.sys": 0.05}
{"size": "small", "cpuid": "2", "compression": "highest", "time.real": 0.32, "time.sys": 0.05}
{"size": "small", "cpuid": "3", "compression": "lowest", "time.real": 0.31, "time.sys": 0.05}
{"size": "small", "cpuid": "3", "compression": "highest", "time.real": 0.33, "time.sys": 0.05}
{"size": "medium", "cpuid": "0", "compression": "lowest", "time.real": 0.62, "time.sys": 0.11}
{"size": "medium", "cpuid": "0", "compression": "highest", "time.real": 0.66, "time.sys": 0.1}
{"size": "medium", "cpuid": "1", "compression": "lowest", "time.real": 0.62, "time.sys": 0.11}
{"size": "medium", "cpuid": "1", "compression": "highest", "time.real": 0.66, "time.sys": 0.1}
{"size": "medium", "cpuid": "2", "compression": "lowest", "time.real": 0.64, "time.sys": 0.11}
{"size": "medium", "cpuid": "2", "compression": "highest", "time.real": 0.65, "time.sys": 0.1}
{"size": "medium", "cpuid": "3", "compression": "lowest", "time.real": 0.67, "time.sys": 0.11}
{"size": "medium", "cpuid": "3", "compression": "highest", "time.real": 0.67, "time.sys": 0.11}
{"size": "large", "cpuid": "0", "compression": "lowest", "time.real": 1.59, "time.sys": 0.27}
{"size": "large", "cpuid": "0", "compression": "highest", "time.real": 1.58, "time.sys": 0.26}
{"size": "large", "cpuid": "1", "compression": "lowest", "time.real": 1.59, "time.sys": 0.28}
{"size": "large", "cpuid": "1", "compression": "highest", "time.real": 1.6, "time.sys": 0.27}
{"size": "large", "cpuid": "2", "compression": "lowest", "time.real": 1.54, "time.sys": 0.38}
{"size": "large", "cpuid": "2", "compression": "highest", "time.real": 1.69, "time.sys": 0.26}
{"size": "large", "cpuid": "3", "compression": "lowest", "time.real": 1.54, "time.sys": 0.27}
{"size": "large", "cpuid": "3", "compression": "highest", "time.real": 1.59, "time.sys": 0.27}
```

These above results can be displayed with `yuclid plot`, e.g.:
```
yuclid plot results.json -x compression
yuclid plot results.json -x size -z cpuid
```
Interact with the plot using arrow keys to move around dimensions and number keys to change the metric!

## Advanced Example

The following is a template showing how to track metrics of a program compiled with different compilers, running with a different number of threads and customize the input based on how many threads are used.

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

## Plot API

`yuclid plot` can be used directly on your pyplot canvas. The command `yuclid plot results.json -x size -z cpuid` can be emulated in a more customizable script, e.g.:

```python
import yuclid.plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# just like the CLI
cli_args = [
  "results.json",
  "-x",
  "size",
  "-z",
  "cpuid"
]
df = yuclid.plot.draw(fig, ax, cli_args)
plt.show()
```

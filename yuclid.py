from datetime import datetime
import itertools
import json
import pandas as pd
import sys
import subprocess
import re

class LogLevel:
    INFO = 1
    WARNING = 2
    ERROR = 3

def message(level, *args, **kwargs):
    timestamp = "{:%Y-%m %d-%H:%M}".format(datetime.now())
    log_prefix = {
        LogLevel.INFO:    "[INFO]",
        LogLevel.WARNING: "[WARNING]",
        LogLevel.ERROR:   "[ERROR]",
    }.get(level, "[UNKNOWN]")
    prefix = f"yuclid {now} {log_prefix}:"
    print(prefix, *args, **kwargs)

def substitute_point_vars(x, point, point_id):
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\}"
    y = re.sub(pattern, lambda m: point[m.group(1)]["name"], x)
    pattern = r"\$\{yuclid\.\#\}"
    y = re.sub(pattern, lambda m: point_id, y)
    return y

def substitute_global_vars(x):
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\.values\}"
    y = re.sub(pattern, lambda m: " ".join(values[m.group(1)]), x)
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\.names\}"
    y = re.sub(pattern, lambda m: " ".join(names[m.group(1)]), y)
    return y

with open("yuclid.json") as f:
    data = json.load(f)

now = "{:%Y%m%d-%H%M}".format(datetime.now())
unfold = True
verbose_results = False

space = dict()
for key, values in data["space"].items():
    if key.endswith(":py"):
        name = key.split(":")[-2]
        space[name] = [{"name": str(x), "value": x} for x in eval(values)]
    else:
        space[key] = []
        for x in values:
            if isinstance(x, str) or isinstance(x, int) or isinstance(x, float):
                space[key].append({"name": str(x), "value": x})
                pass
            elif isinstance(x, dict):
                if "value" in x:
                    space[key].append({"name": str(x.get("name", x["value"])),
                                       "value": x["value"]})

values = {key: [x["value"] for x in space[key]] for key in space}
names = {key: x["value"] for x in space[key]}

setup = []
for command in data["setup"]:
    command = substitute_global_vars(command)
    setup.append(command)

order = list(space.keys())
for key in data.get("order", []):
    order.append(order.pop(order.index(key)))

columns = list(space.keys()) + list(data["metrics"].keys())

def run_setup():
    for command in setup:
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            message(LogLevel.ERROR, f"Command failed with return code {result.returncode}")
        pass

def run_trials(f):
    global point, result
    ordered_space = [space[x] for x in order]
    for i, configuration in enumerate(itertools.product(*ordered_space)):
        point = {key: x for key, x in zip(order, configuration)}
        point_id = f"yuclid.{i:08x}.tmp"
        command = substitute_global_vars(data["trial"])
        command = substitute_point_vars(data["trial"], point, point_id)
        cmd_result = subprocess.run(command, shell=True,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        if cmd_result.returncode != 0:
            message(LogLevel.ERROR,
                    f"{point} failed experiment (rc: {cmd_result.returncode})")
        if verbose_results:
            result = point
        else:
            result = {k: x["value"] for k, x in point.items()}
        max_n_metric_values = None
        mvalues = dict()
        for metric, command in data["metrics"].items():
            command = substitute_global_vars(command)
            command = substitute_point_vars(command, point, point_id)
            cmd_result = subprocess.run(command, shell=True, text=True,
                                 capture_output=True)
            if cmd_result.returncode != 0:
                message(LogLevel.ERROR,
                        f"{point} failed metric '{metric}' (rc: {cmd_result.returncode})")
            cmd_lines = cmd_result.stdout.strip().split("\n")
            mvalues[metric] = [float(line) for line in cmd_lines]
        mvalues_df = pd.DataFrame.from_dict(mvalues, orient="index").transpose()
        mvalues = mvalues_df.to_dict(orient="list")
        result.update(mvalues)
        f.write(json.dumps(result) + "\n")
        message(LogLevel.INFO, *result.values())
        f.flush()

output_file = f"trials.{now}.json"
with open("result.json", "a") as f:
    run_trials(f)

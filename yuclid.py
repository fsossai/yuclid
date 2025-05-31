from datetime import datetime
import pandas as pd
import subprocess
import itertools
import json
import sys
import re
import os

class LogLevel:
    INFO = 1
    WARNING = 2
    ERROR = 3

def message(level, *args, **kwargs):
    timestamp = "{:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    log_prefix = {
        LogLevel.INFO:    "INFO",
        LogLevel.WARNING: "WARNING",
        LogLevel.ERROR:   "ERROR",
    }.get(level, "UNKNOWN")
    prefix = f"{timestamp} yuclid.{log_prefix}:"
    print(prefix, *args, **kwargs)

def substitute_point_vars(x, point, point_id):
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\}"
    y = re.sub(pattern, lambda m: point[m.group(1)]["value"], x)
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
fold = True
verbose_results = False

space = dict()
for key, values in data["space"].items():
    if key.endswith(":py"):
        name = key.split(":")[-2]
        space[name] = [{"name": x, "value": x} for x in eval(values)]
    else:
        space[key] = []
        for x in values:
            if isinstance(x, str) or isinstance(x, int) or isinstance(x, float):
                space[key].append({"name": str(x), "value": x})
                pass
            elif isinstance(x, dict):
                if "value" in x:
                    space[key].append({"name": x.get("name", x["value"]),
                                       "value": x["value"]})

values = {key: [x["value"] for x in space[key]] for key in space}
names = {key: x["value"] for x in space[key]}

env = {k: str(v) for k, v in data["env"].items()}

setup = []
for command in data["setup"]:
    command = substitute_global_vars(command)
    setup.append(command)

order = list(space.keys())
for key in data.get("order", []):
    order.append(order.pop(order.index(key)))

columns = list(space.keys()) + list(data["metrics"].keys())

def run_setup():
    message(LogLevel.INFO, "starting setup")
    errors = False
    for command in setup:
        result = subprocess.run(command, shell=True, env=env)
        if result.returncode != 0:
            errors = True
            message(LogLevel.ERROR, f"'{command}': failed setup {result.returncode}")
    if errors:
        message(LogLevel.WARNING, "errors have occurred during setup")
    message(LogLevel.INFO, "setup completed")

def point_to_string(point):
    return ".".join([str(x["name"]) for x in point.values()])

def metrics_to_string(mvalues):
    return " ".join([f"{m}={v}" for m, v in mvalues.items()])

def run_trials(f):
    global final_metrics, point, mvalues_df
    ordered_space = [space[x] for x in order]
    for i, configuration in enumerate(itertools.product(*ordered_space)):
        point = {key: x for key, x in zip(order, configuration)}
        point_id = f"yuclid.{i:08x}.tmp"
        command = substitute_global_vars(data["trial"])
        command = substitute_point_vars(data["trial"], point, point_id)
        cmd_result = subprocess.run(command, shell=True, env=env,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        if cmd_result.returncode != 0:
            message(LogLevel.ERROR,
                    "{} failed experiment (returned {})".format(
                        point_to_string(point),
                        cmd_result.returncode))
        max_n_metric_values = None
        mvalues = dict()
        for metric, command in data["metrics"].items():
            command = substitute_global_vars(command)
            command = substitute_point_vars(command, point, point_id)
            cmd_result = subprocess.run(command, shell=True, text=True,
                                        capture_output=True, env=env)
            if cmd_result.returncode != 0:
                message(LogLevel.ERROR, "{} failed metric '{}' (returned {})".format(
                    point_to_string(point), metric, cmd_result.returncode))
            cmd_lines = cmd_result.stdout.strip().split("\n")
            mvalues[metric] = [float(line) for line in cmd_lines]
        mvalues_df = pd.DataFrame.from_dict(mvalues, orient="index").transpose()
        if verbose_results:
            result = point
        else:
            result = {k: x["name"] for k, x in point.items()}
        if fold:
            result.update(mvalues_df.to_dict(orient="list"))
            f.write(json.dumps(result) + "\n")
        else:
            for record in mvalues_df.to_dict(orient="records"):
                result.update(record)
                f.write(json.dumps(result) + "\n")
        message(LogLevel.INFO, point_to_string(point), metrics_to_string(mvalues))
        f.flush()

output_file = f"trials.{now}.json"
run_setup()
with open("result.json", "a") as f:
    run_trials(f)

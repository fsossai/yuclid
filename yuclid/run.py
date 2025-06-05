from yuclid.log import LogLevel, TextColors, report
from datetime import datetime
import pandas as pd
import subprocess
import itertools
import argparse
import random
import string
import json
import sys
import re
import os


def substitute_point_vars(ctx, x, point, point_id):
    args = ctx["args"]
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\}"
    y = re.sub(pattern, lambda m: str(point[m.group(1)]["value"]), x)
    pattern = r"\$\{yuclid\.\@\}"
    y = re.sub(pattern, lambda m: f"{args.temp_directory}/{point_id}", y)
    return y


def substitute_global_vars(ctx, x):
    subspace_values = ctx["subspace_values"]
    subspace_names = ctx["subspace_names"]
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\.values\}"
    y = re.sub(pattern, lambda m: " ".join(subspace_values[m.group(1)]), x)
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\.names\}"
    y = re.sub(pattern, lambda m: " ".join(subspace_names[m.group(1)]), y)
    return y


def read_configurations(ctx):
    args = ctx["args"]
    data = {
        "env": dict(),
        "setup": [],
        "space": dict(),
        "trial": [],
        "metrics": dict(),
        "presets": dict(),
        "order": [],
    }

    for file in args.inputs:
        with open(file, "r") as f:
            current = normalize_data(ctx, json.load(f))
            for key, val in current.items():
                if isinstance(data[key], list):
                    data[key].extend(val)
                elif isinstance(data[key], dict):
                    if key == "space":
                        for subkey, subval in val.items():
                            if data[key].get(subkey) is None:
                                data[key][subkey] = subval
                            else:
                                data[key].setdefault(subkey, []).extend(subval)
                    else:
                        data[key].update(val)

    order_seen = set(data.get("order", []))
    data["order"] = [
        x for x in data["order"] if not (x in order_seen or order_seen.add(x))
    ]
    ctx["data"] = data


def build_environment(ctx):
    data = ctx["data"]
    env = os.environ.copy()
    env.update({k: str(v) for k, v in data["env"].items()})
    ctx["env"] = env


def overwrite_configuration(ctx):
    args = ctx["args"]
    subspace = ctx["subspace"]
    if args.select is not None:
        new_values = dict(pair.split("=") for pair in args.select)
        for k, values in new_values.items():
            selection = []
            if subspace[k] is None:
                selection = [{"name": str(x), "value": x} for x in values.split(",")]
            else:
                valid = {str(x["name"]): x for x in subspace[k]}
                for current in values.split(","):
                    if current in valid.keys():
                        selection.append(valid[current])
            if len(selection) == 0:
                report(LogLevel.FATAL, "invalid value", values)
            subspace[k] = selection
    ctx["subspace"] = subspace


def normalize_command(cmd):
    if isinstance(cmd, str):
        return cmd
    elif isinstance(cmd, list):
        return " ".join(cmd)
    else:
        raise ValueError(f"invalid command type {type(cmd)}")


def normalize_command_list(cl):
    normalized = []
    if isinstance(cl, str):
        normalized = [cl]
    elif isinstance(cl, list):
        for cmd in cl:
            normalized.append(normalize_command(cmd))
    return normalized


def normalize_data(ctx, json_data):
    normalized = json_data.copy()

    space = dict()
    for key, values in json_data.get("space", dict()).items():
        if key.endswith(":py"):
            name = key.split(":")[-2]
            space[name] = [{"name": x, "value": x} for x in eval(values)]
        elif values is not None:
            space[key] = []
            for x in values:
                if isinstance(x, (str, int, float)):
                    space[key].append({"name": str(x), "value": x})
                elif isinstance(x, dict):
                    if "value" in x:
                        space[key].append(
                            {"name": x.get("name", x["value"]), "value": x["value"]}
                        )
        else:
            space[key] = None

    metrics = dict()
    for key, value in json_data.get("metrics", dict()).items():
        metrics[key] = normalize_command(value)

    normalized["space"] = space
    normalized["trial"] = normalize_command_list(json_data.get("trial", []))
    normalized["setup"] = normalize_command_list(json_data.get("setup", []))
    normalized["metrics"] = metrics

    return normalized


def build_space(ctx):
    data = ctx["data"]
    space = data["space"]
    defined_space = {k: v for k, v in space.items() if v is not None}
    defined_space_values = {
        key: [x["value"] for x in space[key]] for key in defined_space
    }
    defined_space_names = {
        key: [x["name"] for x in space[key]] for key in defined_space
    }
    undefined_space_values = {key: [] for key in space if space[key] is None}
    undefined_space_names = {key: [] for key in space if space[key] is None}
    space_values = {**defined_space_values, **undefined_space_values}
    space_names = {**defined_space_names, **undefined_space_names}
    ctx["space"] = space
    ctx["space_values"] = space_values
    ctx["space_names"] = space_names


def define_order(ctx):
    args = ctx["args"]
    data = ctx["data"]
    space = ctx["space"]
    if args.order is None:
        desired = data.get("order", [])
    else:
        desired = args.order.split(",")
    order = list(space.keys())
    for k in desired:
        order.append(order.pop(order.index(k)))
    wrong = [k for k in order if k not in space.keys()]
    if len(wrong) > 0:
        report(
            ctx,
            LogLevel.FATAL,
            "specified order contains invalid dimensions",
            ",".join(wrong),
        )
    ctx["order"] = order


def build_subspace(ctx):
    space = ctx["space"]
    selected_presets = ctx["selected_presets"]
    subspace = dict()
    for key, values in space.items():
        subvalues = []
        relevant_presets = {
            pname: pdims for pname, pdims in selected_presets.items() if key in pdims
        }
        if len(relevant_presets) == 0:
            subvalues = values
        elif len(relevant_presets) == 1:
            pname, pdims = next(iter(relevant_presets.items()))
            if values is None:
                subvalues = [{"name": str(x), "value": x} for x in pdims[key]]
            else:
                vmap = {x["name"]: x for x in values}
                subvalues = [vmap[n] for n in pdims[key] if n in vmap]
        else:
            report(
                ctx,
                LogLevel.FATAL,
                f"dimension '{key}' conflicts in presets",
                ", ".join([pname for pname in relevant_presets]),
            )
        subspace[key] = subvalues
    ctx["subspace"] = subspace


def run_setup(ctx):
    args = ctx["args"]
    data = ctx["data"]
    env = ctx["env"]
    if args.dry_run:
        return
    setup = []
    for command in data["setup"]:
        command = substitute_global_vars(ctx, command)
        setup.append(command)
    report(LogLevel.INFO, "starting setup")
    errors = False
    for command in setup:
        result = subprocess.run(command, shell=True, env=env)
        if result.returncode != 0:
            errors = True
            report(
                LogLevel.ERROR,
                "setup",
                f"'{command}'",
                f"failed (code {result.returncode})",
            )
    if errors:
        report(LogLevel.WARNING, "errors have occurred during setup")
    report(LogLevel.INFO, "setup completed")


def point_to_string(point):
    return ".".join([str(x["name"]) for x in point.values()])


def metrics_to_string(mvalues):
    from yuclid.log import _state

    color = _state["color"]
    return " ".join([f"{color.bold}{m}{color.none}={v}" for m, v in mvalues.items()])


def get_progress(i, subspace_size):
    return "[{}/{}]".format(i, subspace_size)


def run_trial(ctx, f, i, configuration):
    args = ctx["args"]
    env = ctx["env"]
    data = ctx["data"]
    order = ctx["order"]
    trial = ctx["trial"]
    point = {key: x for key, x in zip(order, configuration)}
    rand_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    point_id = "{}.{}.tmp".format(rand_str, point_to_string(point))
    report(
        LogLevel.INFO,
        get_progress(i, ctx["subspace_size"]),
        point_to_string(point),
        "started",
    )
    command = substitute_global_vars(ctx, trial)
    command = substitute_point_vars(ctx, command, point, point_id)
    cmd_result = subprocess.run(
        command,
        shell=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if cmd_result.returncode != 0:
        report(
            LogLevel.ERROR,
            point_to_string(point),
            f"failed experiment (code {cmd_result.returncode})",
        )
    mvalues = dict()
    for metric, command in data["metrics"].items():
        command = substitute_global_vars(ctx, command)
        command = substitute_point_vars(ctx, command, point, point_id)
        cmd_result = subprocess.run(
            command, shell=True, universal_newlines=True, capture_output=True, env=env
        )
        if cmd_result.returncode != 0:
            report(
                LogLevel.ERROR,
                point_to_string(point),
                "failed metric '{}' (code {})".format(metric, cmd_result.returncode),
            )
        cmd_lines = cmd_result.stdout.strip().split("\n")
        mvalues[metric] = [float(line) for line in cmd_lines]
    mvalues_df = pd.DataFrame.from_dict(mvalues, orient="index").transpose()
    if not args.fold:
        NaNs = mvalues_df.columns[mvalues_df.isnull().any()]
        if len(NaNs) > 0:
            report(
                LogLevel.WARNING,
                "the following metrics generated some NaNs",
                " ".join(list(NaNs)),
            )

    if args.verbose_data:
        result = point
    else:
        result = {k: x["name"] for k, x in point.items()}
    if args.fold:
        result.update(mvalues_df.to_dict(orient="list"))
        f.write(json.dumps(result) + "\n")
    else:
        for record in mvalues_df.to_dict(orient="records"):
            result.update(record)
            f.write(json.dumps(result) + "\n")
    report(LogLevel.INFO, "obtained", metrics_to_string(mvalues))
    report(
        LogLevel.INFO,
        get_progress(i + 1, ctx["subspace_size"]),
        point_to_string(point),
        "completed",
    )
    f.flush()


def run_trials(ctx):
    args = ctx["args"]
    data = ctx["data"]
    order = ctx["order"]
    subspace = ctx["subspace"]
    ordered_space = [subspace[x] for x in order]

    if "trial" not in data:
        report(LogLevel.FATAL, "missing 'trial' command")
    if isinstance(data["trial"], str):
        trial = data["trial"]
    elif isinstance(data["trial"], list):
        trial = " ".join(data["trial"])
    ctx["trial"] = trial

    if args.dry_run:
        for i, configuration in enumerate(itertools.product(*ordered_space)):
            point = {key: x for key, x in zip(order, configuration)}
            report(
                LogLevel.INFO,
                get_progress(i + 1, ctx["subspace_size"]),
                "dry run",
                point_to_string(point),
            )
    else:
        report(LogLevel.INFO, f"writing to '{args.output}'")
        with open(args.output, "a") as f:
            for i, configuration in enumerate(itertools.product(*ordered_space)):
                run_trial(ctx, f, i, configuration)


def validate_presets(ctx):
    args = ctx["args"]
    data = ctx["data"]
    space = ctx["space"]
    space_names = ctx["space_names"]
    space_values = ctx["space_values"]
    # normalization
    presets_old = data.get("presets", dict())
    presets = dict()
    for pname, pspace in presets_old.items():
        presets[pname] = dict()
        for k, values in pspace.items():
            if k.endswith(":py"):
                k = k.split(":py")[-2]
                if isinstance(values, str):
                    presets[pname][k] = eval(values)
                else:
                    report(
                        ctx,
                        LogLevel.FATAL,
                        "pythonic dimensions must be strings",
                        f"in '{k}' in '{pname}'",
                    )
            elif not isinstance(values, list):
                presets[pname][k] = [values]
            else:
                presets[pname][k] = values

    for pname, pspace in presets.items():
        for k, values in pspace.items():
            if k not in space:
                report(LogLevel.FATAL, "preset dimension not in space", k)
            new_values = []
            wrong = []
            for v in values:
                if isinstance(v, str) and "*" in v:
                    if space[k] is None:
                        report(
                            ctx,
                            LogLevel.FATAL,
                            "regex cannot be used on undefined dimensions",
                            k,
                        )
                    else:
                        pattern = "^" + re.escape(v).replace("\\*", ".*") + "$"
                        regex = re.compile(pattern)
                        new_values += [n for n in space_names[k] if regex.match(n)]
                elif v not in space_values[k] and space[k] is not None:
                    wrong.append(str(v))
                else:
                    new_values.append(v)
            if len(wrong) > 0:
                report(
                    ctx,
                    LogLevel.FATAL,
                    f"unknown values in preset '{pname}'",
                    ", ".join(wrong),
                )
            presets[pname][k] = new_values

    for pname, pspace in presets.items():
        for k, v in pspace.items():
            if len(v) == 0:
                report(LogLevel.ERROR, f"empty dimension in preset '{pname}'", k)

    if args.presets is None:
        selected_presets = dict()
    else:
        selected_presets = dict()
        for p in args.presets.split(","):
            if p not in presets:
                report(LogLevel.FATAL, "unknown preset", p)
            else:
                selected_presets[p] = presets[p]
    ctx["presets"] = presets
    ctx["selected_presets"] = selected_presets


def validate_subspace(ctx):
    subspace = ctx["subspace"]
    undefined = [k for k, v in subspace.items() if v is None]
    if len(undefined) > 0:
        report(LogLevel.FATAL, "dimensions undefined", ", ".join(undefined))
    ctx["subspace_size"] = pd.Series([len(v) for k, v in subspace.items()]).prod()
    ctx["subspace_values"] = {
        key: [x["value"] for x in subspace[key]] for key in subspace
    }
    ctx["subspace_names"] = {
        key: [x["name"] for x in subspace[key]] for key in subspace
    }


def validate_args(ctx):
    args = ctx["args"]
    now = "{:%Y%m%d-%H%M}".format(datetime.now())
    filename = f"trials.{now}.json"

    if args.output is None and args.output_dir is None:
        ctx["output"] = filename
    elif args.output is not None and args.output_dir is not None:
        report(LogLevel.FATAL, "either --output or --output-dir must be specified")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        ctx["output"] = os.path.join(args.output_dir, filename)
    else:
        ctx["output"] = args.output
    for file in args.inputs:
        if not os.path.isfile(file):
            report(LogLevel.FATAL, f"'{file}' does not exist")
    os.makedirs(args.temp_directory, exist_ok=True)
    report(LogLevel.INFO, "input configurations", ", ".join(args.inputs))
    report(LogLevel.INFO, "output data", ctx["output"])
    report(LogLevel.INFO, "temp directory", f"'{args.temp_directory}'")


def launch(args):
    ctx = {"args": args, "color": TextColors()}
    validate_args(ctx)
    read_configurations(ctx)
    build_environment(ctx)
    build_space(ctx)
    validate_presets(ctx)
    build_subspace(ctx)
    overwrite_configuration(ctx)
    validate_subspace(ctx)
    define_order(ctx)
    run_setup(ctx)
    run_trials(ctx)

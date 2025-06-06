from yuclid.log import LogLevel, report
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
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\}"
    y = re.sub(pattern, lambda m: str(point[m.group(1)]["value"]), x)
    pattern = r"\$\{yuclid\.\@\}"
    y = re.sub(pattern, lambda m: f"{point_id}", y)
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
            current = normalize_data(json.load(f))
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

            order = data.get("order", []) + current.get("order", [])
            data["order"] = remove_duplicates(order)

    ctx["data"] = data


def remove_duplicates(items):
    seen = list()
    return [x for x in items if not (x in seen or seen.append(x))]


def build_environment(ctx):
    if ctx["args"].dry_run:
        for key, value in ctx["data"]["env"].items():
            report(LogLevel.INFO, "dry env", f'{key}="{value}"')
        ctx["env"] = dict()
    else:
        data = ctx["data"]
        resolved_env = os.environ.copy()
        for k, v in data["env"].items():
            expanded = subprocess.run(
                f'echo "{v}"',
                env=resolved_env,
                capture_output=True,
                text=True,
                shell=True,
            ).stdout.strip()
            resolved_env[k] = expanded
        ctx["env"] = resolved_env


def overwrite_configuration(ctx):
    args = ctx["args"]
    subspace = ctx["subspace"]
    if args.select is not None:
        new_values = dict(pair.split("=") for pair in args.select)
        for k, values in new_values.items():
            selection = []
            if subspace[k] is None:
                selection = [normalize_point(x) for x in values.split(",")]
            else:
                valid = {str(x["name"]): x for x in subspace[k]}
                for current in values.split(","):
                    if current in valid.keys():
                        selection.append(valid[current])
            if len(selection) == 0:
                available = ", ".join(
                    [str(x["name"]) for x in subspace[k]]
                    if subspace[k] is not None
                    else []
                )
                hint = "pick from the following values: {}".format(available)
                report(LogLevel.FATAL, "invalid value", values, hint=hint)
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


def normalize_point(x):
    normalized = None
    if isinstance(x, (str, int, float)):
        normalized = {"name": str(x), "value": x, "group": 0}
    elif isinstance(x, dict):
        if "value" in x:
            normalized = {
                "name": x.get("name", x["value"]),
                "value": x["value"],
                "group": x.get("group", 0),
            }
    return normalized


def normalize_data(json_data):
    normalized = json_data.copy()

    space = dict()
    for key, values in json_data.get("space", dict()).items():
        if key.endswith(":py"):
            name = key.split(":")[-2]
            space[name] = [{"name": x, "value": x, "group": 0} for x in eval(values)]
        elif values is not None:
            space[key] = []
            for x in values:
                space[key].append(normalize_point(x))
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

    available = space.keys()
    user_specified = [] if args.order is None else args.order.split(",")
    in_config = data.get("order", [])
    wrong = [k for k in set(user_specified + in_config) if k not in available]
    if len(wrong) > 0:
        hint = "available values: {}".format(", ".join(available))
        report(
            LogLevel.FATAL,
            "invalid order values",
            ", ".join(wrong),
            hint=hint,
        )

    def reorder(desired):
        for k in desired:
            order.append(order.pop(order.index(k)))

    order = list(space.keys())
    reorder(in_config)
    reorder(user_specified)

    ctx["order"] = order


def build_subspace(ctx):
    space = ctx["space"]
    subspace = dict()
    preset = ctx["presets"][ctx["current_preset"]]
    for key, values in space.items():
        if key in preset:
            subvalues = []
            if values is None:
                subvalues = [normalize_point(x) for x in preset[key]]
            else:
                vmap = {x["name"]: x for x in values}
                subvalues = [vmap[n] for n in preset[key] if n in vmap]
            subspace[key] = subvalues
        else:
            subspace[key] = values
    ctx["subspace"] = subspace


def run_setup(ctx):
    args = ctx["args"]
    data = ctx["data"]
    setup = []
    for command in data["setup"]:
        command = substitute_global_vars(ctx, command)
        setup.append(command)

    if args.dry_run:
        report(LogLevel.INFO, "starting dry setup")
    else:
        report(LogLevel.INFO, "starting setup")
    errors = False
    for command in setup:
        if args.dry_run:
            report(LogLevel.INFO, "dry run", command)
        else:
            result = subprocess.run(
                command,
                shell=True,
                universal_newlines=True,
                capture_output=False,
                env=ctx["env"],
            )
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
        report(LogLevel.INFO, "setup failed")
    if args.dry_run:
        report(LogLevel.INFO, "dry setup completed")
    else:
        report(LogLevel.INFO, "setup completed")


def point_to_string(point):
    return ".".join([str(x["name"]) for x in point.values()])


def metrics_to_string(mvalues):
    return " ".join([f"{m}={v}" for m, v in mvalues.items()])


def get_progress(i, subspace_size):
    return "[{}/{}]".format(i, subspace_size)


def run_trial(ctx, f, i, configuration):
    args = ctx["args"]
    env = ctx["env"]
    data = ctx["data"]
    order = ctx["order"]
    trial = ctx["trial"]
    point = {key: x for key, x in zip(order, configuration)}

    point_id = os.path.join(
        args.temp_dir,
        "{}.{}.tmp".format(ctx["random_key"], point_to_string(point)),
    )
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
        if os.path.exists(point_id):
            hint = "try `cat {}` for more information".format(point_id)
        else:
            hint = None
        report(
            LogLevel.ERROR,
            point_to_string(point),
            f"failed trials (code {cmd_result.returncode})",
            hint=hint,
        )
    mvalues = dict()
    for metric, command in data["metrics"].items():
        command = substitute_global_vars(ctx, command)
        command = substitute_point_vars(ctx, command, point, point_id)
        cmd_result = subprocess.run(
            command, shell=True, universal_newlines=True, capture_output=True, env=env
        )
        if cmd_result.returncode != 0:
            hint = "the command '{}' produced the following output:\n{}".format(
                command,
                cmd_result.stdout.strip(),
            )
            report(
                LogLevel.ERROR,
                point_to_string(point),
                "failed metric '{}' (code {})".format(metric, cmd_result.returncode),
                hint=hint,
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
        get_progress(i, ctx["subspace_size"]),
        point_to_string(point),
        "completed",
    )
    f.flush()


def trim_groups(subspace):
    return subspace


def compatible_groups(configuration):
    non_neutral_groups = [x["group"] for x in configuration if x["group"] != 0]
    return len(set(non_neutral_groups)) == 1


def run_trials(ctx):
    args = ctx["args"]
    data = ctx["data"]
    order = ctx["order"]
    subspace = ctx["subspace"]

    if len(data.get("trial", [])) == 0:
        report(LogLevel.FATAL, "missing 'trial' command")
    if isinstance(data.get("trial"), str):
        trial = data["trial"]
    elif isinstance(data.get("trial"), list):
        trial = " ".join(data["trial"])
    else:
        hint = "try string or list of strings"
        report(LogLevel.FATAL, "wrong format for 'trial'", hint=hint)
    ctx["trial"] = trial

    if args.dry_run:
        for i, configuration in enumerate(ctx["subspace_points"], start=1):
            point = {key: x for key, x in zip(order, configuration)}
            if compatible_groups(configuration):
                report(
                    LogLevel.INFO,
                    get_progress(i, ctx["subspace_size"]),
                    "dry run",
                    point_to_string(point),
                )
    else:
        with open(ctx["output"], "a") as f:
            for i, configuration in enumerate(ctx["subspace_points"], start=1):
                run_trial(ctx, f, i, configuration)
                f.flush()


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
                hint = "available dimensions: {}".format(", ".join(space.keys()))
                report(LogLevel.FATAL, "preset dimension not in space", k, hint=hint)
            new_values = []
            wrong = []
            for v in values:
                if isinstance(v, str) and "*" in v:
                    if space[k] is None:
                        report(
                            LogLevel.FATAL,
                            "regex cannot be used on undefined dimensions",
                            k,
                        )
                    else:
                        pattern = "^" + re.escape(v).replace("\\*", ".*") + "$"
                        regex = re.compile(pattern)
                        new_values += [n for n in space_names[k] if regex.match(n)]
                elif v not in space_names[k] and space[k] is not None:
                    wrong.append(str(v))
                else:
                    new_values.append(v)

            if len(wrong) > 0:
                hint = "available names: {}".format(", ".join(space_names[k]))
                report(
                    LogLevel.FATAL,
                    f"unknown name in preset '{pname}'",
                    ", ".join(wrong),
                    hint=hint,
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
                hint = "available presets: {}".format(", ".join(presets.keys()))
                report(LogLevel.FATAL, "unknown preset", p, hint=hint)
            else:
                selected_presets[p] = presets[p]
    ctx["presets"] = presets
    ctx["selected_presets"] = selected_presets


def validate_subspace(ctx):
    subspace = ctx["subspace"]
    order = ctx["order"]
    ordered_subspace = [subspace[x] for x in order]
    undefined = [k for k, v in subspace.items() if v is None]
    if len(undefined) > 0:
        hint = "define dimensions with presets or select them with --select. "
        hint += "E.g. --select {}=value1,value2".format(undefined[0])
        report(LogLevel.FATAL, "dimensions undefined", ", ".join(undefined), hint=hint)

    ctx["subspace_points"] = []
    for point in itertools.product(*ordered_subspace):
        if compatible_groups(point):
            ctx["subspace_points"].append(point)
    ctx["subspace_size"] = len(ctx["subspace_points"])

    ctx["subspace_values"] = {
        key: [x["value"] for x in subspace[key]] for key in subspace
    }
    ctx["subspace_names"] = {
        key: [x["name"] for x in subspace[key]] for key in subspace
    }

    # checking group compatibility
    for key, values in subspace.items():
        groups = {x["group"] for x in values}
        if len(groups) > 1 and 0 in groups:
            hint = "items with neutral group (i.e. 0) should be all or none"
            report(LogLevel.WARNING, "unusual group configuration", key, hint=hint)


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
    os.makedirs(args.temp_dir, exist_ok=True)
    ctx["random_key"] = "".join(
        random.choices(string.ascii_letters + string.digits, k=8)
    )
    report(LogLevel.INFO, "random key", ctx["random_key"])
    report(LogLevel.INFO, "working directory", os.getcwd())
    report(LogLevel.INFO, "input configurations", ", ".join(args.inputs))
    report(LogLevel.INFO, "output data", ctx["output"])
    report(LogLevel.INFO, "temp directory", args.temp_dir)


def launch(args):
    ctx = {"args": args}
    validate_args(ctx)
    read_configurations(ctx)
    build_environment(ctx)
    build_space(ctx)
    define_order(ctx)
    validate_presets(ctx)

    if len(ctx["selected_presets"]) > 0:
        for preset_name in ctx["selected_presets"]:
            ctx["current_preset"] = preset_name
            report(LogLevel.INFO, "loading preset", preset_name)
            build_subspace(ctx)
            overwrite_configuration(ctx)
            validate_subspace(ctx)
            run_setup(ctx)
            run_trials(ctx)
            report(LogLevel.INFO, "completed preset", preset_name)
    else:
        ctx["subspace"] = ctx["space"].copy()
        overwrite_configuration(ctx)
        validate_subspace(ctx)
        run_setup(ctx)
        run_trials(ctx)

    report(LogLevel.INFO, "finished")
    if not args.dry_run:
        y_axis = ctx["data"]["metrics"].keys()
        hint = "use `yuclid plot {} -y {}` to analyze the results".format(
            ctx["output"], ",".join(y_axis)
        )
        report(LogLevel.INFO, "output data written to", ctx["output"], hint=hint)

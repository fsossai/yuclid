from yuclid.log import LogLevel, report
from datetime import datetime
import concurrent.futures
import pandas as pd
import subprocess
import itertools
import threading
import argparse
import random
import string
import json
import sys
import re
import os


def substitute_point_vars(x, point, point_id):
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\}"
    y = re.sub(pattern, lambda m: str(point[m.group(1)]["value"]), x)
    if point_id is not None:
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


def get_yvar_pattern():
    return r"\$\{yuclid\.([a-zA-Z0-9_@]+)\}"


def validate_yvars_in_env(ctx):
    for key, value in ctx["data"]["env"].items():
        if re.search(get_yvar_pattern(), value):
            hint = (
                "maybe you meant ${{yuclid.{}.names}} or ${{yuclid.{}.values}}?".format(
                    key, key
                )
            )
            report(
                LogLevel.FATAL,
                f"cannot use yuclid point variables in env",
                value,
                hint=hint,
            )


def validate_vars_in_setup(ctx):
    data = ctx["data"]
    setup = data["setup"]

    # global setup
    for command in setup["global"]:
        # match ${yuclid.<name>}
        # for all matches, check if the name is in on_dims
        names = re.findall(get_yvar_pattern(), command)
        for name in names:
            hint = (
                "maybe you meant ${{yuclid.{}.names}} or ${{yuclid.{}.values}}?".format(
                    name, name
                )
            )
            report(
                LogLevel.FATAL,
                f"cannot use yuclid point variables in global setup",
                command,
                hint=hint,
            )

    # point setup
    for point_item in setup["point"]:
        on_dims = point_item["on"] or data["space"].keys()
        commands = point_item["commands"]
        for command in commands:
            # match ${yuclid.(<name>|@)}
            pattern = r"\$\{yuclid\.([a-zA-Z0-9_@]+)\}"
            # for all matches, check if the name is in on_dims
            names = re.findall(pattern, command)
            for name in names:
                if name not in on_dims:
                    hint = "available variables: {}".format(
                        ", ".join(["${{yuclid.{}}}".format(d) for d in on_dims])
                    )
                    if name == "@":
                        hint = ". ${yuclid.@} is reserved for trial commands"
                        report(
                            LogLevel.FATAL,
                            f"invalid yuclid variable '{name}' in point setup",
                            command,
                            hint=hint,
                        )


def validate_yvars(ctx):
    validate_yvars_in_env(ctx)
    validate_vars_in_setup(ctx)


def load_json(f):
    try:
        return json.load(f)
    except json.JSONDecodeError as e:
        report(
            LogLevel.FATAL,
            "failed to parse JSON",
            f.name,
            f"at line {e.lineno}, column {e.colno}: {e.msg}",
        )


def read_configurations(ctx):
    args = ctx["args"]
    data = None

    for file in args.inputs:
        with open(file, "r") as f:
            current = normalize_data(load_json(f))
            if data is None:
                data = current
                continue
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


def apply_user_selectors(subspace, selector_pairs):
    selectors = dict(pair.split("=") for pair in selector_pairs)
    for dim, csv_selection in selectors.items():
        selectors = csv_selection.split(",")
        if subspace[dim] is None:
            selection = [normalize_point(x) for x in selectors]
        else:
            selection = []
            valid = {str(x["name"]): x for x in subspace[dim]}
            for selector in selectors:
                if selector in valid.keys():
                    selection.append(valid[selector])
                else:
                    report(
                        LogLevel.ERROR,
                        "invalid selector",
                        selector,
                        hint="available: {}".format(", ".join(valid.keys())),
                    )

            if len(selection) == 0:
                available = (
                    [str(x["name"]) for x in subspace[dim]]
                    if subspace[dim] is not None
                    else []
                )
                if len(available) == 0:
                    hint = None
                else:
                    hint = "pick from the following values: {}".format(
                        ", ".join(available)
                    )
                report(
                    LogLevel.FATAL,
                    "no valid selection for dimension '{}'".format(dim),
                    hint=hint,
                )
        subspace[dim] = selection
    return subspace


def normalize_command(cmd):
    if isinstance(cmd, str):
        return cmd
    elif isinstance(cmd, list):
        return " ".join(cmd)
    else:
        report(LogLevel.FATAL, "command must be a string or a list of strings", cmd)


def normalize_command_list(cl):
    normalized = []
    if isinstance(cl, str):
        normalized = [cl]
    elif isinstance(cl, list):
        for cmd in cl:
            normalized.append(normalize_command(cmd))
    return normalized


def normalize_condition(x):
    if not isinstance(x, str):
        report(LogLevel.FATAL, "condition must be a string", x)
    return x


def normalize_point(x):
    normalized = None
    if isinstance(x, (str, int, float)):
        normalized = {"name": str(x), "value": x, "condition": "True", "setup": []}
    elif isinstance(x, dict):
        if "value" in x:
            normalized = {
                "name": str(x.get("name", x["value"])),
                "value": x["value"],
                "condition": normalize_condition(x.get("condition", "True")),
                "setup": normalize_command_list(x.get("setup", [])),
            }
    elif isinstance(x, list):
        normalized = [normalize_point(item) for item in x]
    return normalized


def normalize_trials(trial):
    if isinstance(trial, str):
        return [{"command": trial, "condition": "True"}]
    elif isinstance(trial, list):
        items = []
        for cmd in trial:
            item = {"command": None, "condition": "True"}
            if isinstance(cmd, str):
                item["command"] = normalize_command(cmd)
            elif isinstance(cmd, dict):
                if "command" not in cmd:
                    report(
                        LogLevel.FATAL, "each trial item must have a 'command' field"
                    )
                    return None
                item["command"] = normalize_command(cmd["command"])
                item["condition"] = cmd.get("condition", "True")
            items.append(item)
        return items
    else:
        report(LogLevel.FATAL, "trial must be a string or a list of strings")
        return None


def normalize_space_values(space):
    normalized = dict()
    for key, values in space.items():
        if key.endswith(":py"):
            name = key.split(":")[-2]
            normalized[name] = normalize_point(eval(values))
        elif values is not None:
            normalized[key] = []
            for x in values:
                normalized[key].append(normalize_point(x))
        else:
            normalized[key] = None
    return normalized


def normalize_data(json_data):
    normalized = {
        "env": dict(),
        "setup": dict(),
        "space": dict(),
        "trials": [],
        "metrics": dict(),
        "presets": dict(),
        "order": [],
    }

    for key in json_data.keys():
        if key in normalized.keys():
            normalized[key] = json_data[key]
        else:
            report(
                LogLevel.WARNING,
                "unknown field in configuration",
                key,
                hint="available fields: {}".format(", ".join(normalized.keys())),
            )

    space = normalize_space_values(json_data.get("space", {}))

    metrics = dict()
    for key, value in json_data.get("metrics", dict()).items():
        metrics[key] = normalize_command(value)

    normalized["space"] = space
    normalized["trials"] = normalize_trials(json_data.get("trials", []))
    normalized["setup"] = normalize_setup(json_data.get("setup", {}))
    normalized["metrics"] = metrics
    normalized["presets"] = json_data.get("presets", dict())

    if len(normalized["trials"]) == 0:
        report(LogLevel.FATAL, "no valid trials found")

    if len(normalized["metrics"]) == 0:
        report(LogLevel.WARNING, "no metrics found. Trials will not be evaluated")

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
    ctx["unfiltered_space"] = space.copy()
    apply_user_selectors(space, ctx["args"].select or [])
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


def build_subspace(ctx, preset_name):
    space = ctx["space"]
    subspace = dict()
    space_names = ctx["space_names"]
    preset_space = ctx["data"]["presets"][preset_name]
    preset = dict()

    for dim, space_items in preset_space.items():
        if dim not in space:
            hint = "available dimensions: {}".format(", ".join(space.keys()))
            report(LogLevel.FATAL, "preset dimension not in space", dim, hint=hint)
        new_items = []
        wrong = []
        for item in space_items:
            if not isinstance(item, (str, int, float)):
                report(
                    LogLevel.FATAL, "preset item must be a string, int or float", item
                )
            if space[dim] is None:
                if isinstance(item, str) and "*" in item:
                    # regex definition
                    report(
                        LogLevel.FATAL,
                        "regex cannot be used on undefined dimensions",
                        dim,
                    )
                elif isinstance(item, (str, int, float)):
                    new_items.append(item)
                else:
                    report(
                        LogLevel.FATAL,
                        "preset item for undefined dimensions must be a string, int or float",
                        item,
                    )
            elif isinstance(item, str) and "*" in item:
                # definition via regex
                pattern = "^" + re.escape(item).replace("\\*", ".*") + "$"
                regex = re.compile(pattern)
                new_items += [n for n in space_names[dim] if regex.match(n)]
            elif str(item) not in space_names[dim]:
                report(
                    LogLevel.ERROR,
                    "unknown name in preset",
                    hint="in order to use '{}' in preset '{}', define it first in the space".format(
                        item, preset_name
                    ),
                )
            else:
                new_items.append(item)

        if len(wrong) > 0:
            hint = "available names: {}".format(", ".join(space_names[dim]))
            report(
                LogLevel.FATAL,
                f"unknown name in preset '{preset_name}'",
                ", ".join(wrong),
                hint=hint,
            )
        preset[dim] = new_items

    for dim, space_items in preset_space.items():
        if len(space_items) == 0:
            report(LogLevel.ERROR, f"empty dimension in preset '{preset_name}'", dim)

    for key, space_items in space.items():
        if key in preset:
            subvalues = []
            if space_items is None:
                subvalues = [normalize_point(x) for x in preset[key]]
            else:
                vmap = {x["name"]: x for x in space_items}
                subvalues = [vmap[n] for n in preset[key] if n in vmap]
            subspace[key] = subvalues
        else:
            subspace[key] = space_items
    apply_user_selectors(subspace, ctx["args"].select or [])
    ctx["subspace"] = subspace


def run_point_setup_item(ctx, item):
    args = ctx["args"]
    data = ctx["data"]
    order = ctx["order"]
    setup = ctx["data"]["setup"]
    on_dims = item["on"]
    commands = item["commands"]
    point_context = get_point_setup_context(ctx, item)

    parallel_space = point_context["parallel_space"]
    sequential_space = point_context["sequential_space"]
    parallel_dims = point_context["parallel_dims"]
    sequential_dims = point_context["sequential_dims"]

    if args.dry_run:
        report(LogLevel.INFO, "starting dry point setup")
    else:
        report(LogLevel.INFO, "starting point setup")

    total_points = ctx["subspace_size"]

    # thread-safe error tracking
    errors_lock = threading.Lock()
    errors = False

    def run_single_point_command(command, configuration):
        nonlocal errors
        gcommand = substitute_global_vars(ctx, command)
        suborder = [d for d in order if d in on_dims]
        point = {key: x for key, x in zip(suborder, configuration)}
        pcommand = substitute_point_vars(gcommand, point, None)

        if not valid_conditions(configuration, suborder):
            return

        if args.dry_run:
            report(
                LogLevel.INFO,
                "dry run",
                pcommand,
            )
        else:
            result = subprocess.run(
                pcommand,
                shell=True,
                universal_newlines=True,
                capture_output=False,
                env=ctx["env"],
            )
            if result.returncode != 0:
                with errors_lock:
                    errors = True
                report(
                    LogLevel.ERROR,
                    "point setup",
                    f"'{command}'",
                    f"failed (code {result.returncode})",
                )

    def run_sequential_points(command, par_config):
        seq_points = list(itertools.product(*sequential_space))
        named_par_config = [(name, x) for name, x in zip(parallel_dims, par_config)]
        if len(sequential_dims) == 0:
            final_config = [x[1] for x in named_par_config]
            run_single_point_command(command, final_config)
            return

        for seq_config in seq_points:
            named_seq_config = [(dim, x) for dim, x in zip(sequential_dims, seq_config)]
            named_ordered_config = sorted(
                named_par_config + named_seq_config, key=lambda x: order.index(x[0])
            )
            final_config = [x[1] for x in named_ordered_config]
            run_single_point_command(command, final_config)

    num_parallel_dims = len(parallel_space)
    if num_parallel_dims == 0:
        max_workers = 1
    else:
        max_workers = min(total_points, os.cpu_count() or 1)
    report(LogLevel.INFO, f"using {max_workers} workers for point setup")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        par_points = list(itertools.product(*parallel_space))

        for i, command in enumerate(commands, start=1):
            if len(parallel_dims) == 0:
                run_sequential_points(command, [])
            else:
                futures = []
                for j, par_config in enumerate(par_points, start=1):
                    future = executor.submit(run_sequential_points, command, par_config)
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    exc = future.exception()
                    if exc is not None:
                        report(LogLevel.ERROR, "point setup", f"failed: {command}")

    if errors:
        report(LogLevel.WARNING, "errors have occurred during point setup")
        report(LogLevel.INFO, "point setup failed")
    if args.dry_run:
        report(LogLevel.INFO, "dry point setup completed")
    else:
        report(LogLevel.INFO, "point setup completed")


def run_point_setup(ctx):
    for item in ctx["data"]["setup"]["point"]:
        if item["on"] is None:
            item["on"] = ctx["data"]["space"].keys()

        # reordering
        item["on"] = [x for x in item["on"] if x in ctx["order"]]
        if len(item["on"]) == 0:
            report(
                LogLevel.WARNING,
                "point setup item has no valid 'on' dimensions. Skipping",
                item,
            )
            continue
        run_point_setup_item(ctx, item)


def run_global_setup(ctx):
    args = ctx["args"]
    data = ctx["data"]
    setup_commands = ctx["data"]["setup"]["global"].copy()

    # gather setup commands from space
    for key, values in ctx["subspace"].items():
        for value in values:
            if len(value["setup"]) > 0:
                for cmd in value["setup"]:
                    setup_commands.append(cmd)

    if args.dry_run:
        report(LogLevel.INFO, "starting dry global setup")
    else:
        report(LogLevel.INFO, "starting global setup")

    errors = False
    for command in setup_commands:
        if args.dry_run:
            report(LogLevel.INFO, "dry run", command)
        else:
            command = substitute_global_vars(ctx, command)
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


def run_setup(ctx):
    run_global_setup(ctx)
    run_point_setup(ctx)


def point_to_string(point):
    return ".".join([str(x["name"]) for x in point.values()])


def metrics_to_string(metric_values):
    return " ".join([f"{m}={v}" for m, v in metric_values.items()])


def get_progress(i, subspace_size):
    return "[{}/{}]".format(i, subspace_size)


def run_point_trials(ctx, f, i, configuration):
    args = ctx["args"]
    env = ctx["env"]
    data = ctx["data"]
    order = ctx["order"]
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

    compatible_trials = [
        trial
        for trial in data["trials"]
        if valid_condition(trial["condition"], configuration, order)
    ]

    if len(compatible_trials) == 0:
        report(LogLevel.WARNING, point_to_string(point), "no compatible trials found")

    for trial in compatible_trials:
        command = substitute_global_vars(ctx, trial["command"])
        command = substitute_point_vars(command, point, point_id)
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

    metric_values = dict()
    for metric, command in data["metrics"].items():
        command = substitute_global_vars(ctx, command)
        command = substitute_point_vars(command, point, point_id)
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
        else:
            cmd_lines = cmd_result.stdout.strip().split("\n")
            metric_values[metric] = [float(line) for line in cmd_lines]

    metric_values_df = pd.DataFrame.from_dict(metric_values, orient="index").transpose()
    if not args.fold:
        NaNs = metric_values_df.columns[metric_values_df.isnull().any()]
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
        result.update(metric_values_df.to_dict(orient="list"))
        f.write(json.dumps(result) + "\n")
    else:
        for record in metric_values_df.to_dict(orient="records"):
            result.update(record)
            f.write(json.dumps(result) + "\n")

    report(LogLevel.INFO, "obtained", metrics_to_string(metric_values))
    report(
        LogLevel.INFO,
        get_progress(i, ctx["subspace_size"]),
        point_to_string(point),
        "completed",
    )
    f.flush()


def valid_conditions(configuration, order):
    point_context = {}
    yuclid = {name: x["value"] for name, x in zip(order, configuration)}
    point_context["yuclid"] = type("Yuclid", (), yuclid)()
    return all(eval(x["condition"], point_context) for x in configuration)


def valid_condition(condition, configuration, order):
    point_context = {}
    yuclid = {name: x["value"] for name, x in zip(order, configuration)}
    point_context["yuclid"] = type("Yuclid", (), yuclid)()
    return eval(condition, point_context)


def validate_points(ctx):
    data = ctx["data"]
    order = ctx["order"]

    # checking if there's at least of compatible trial command for each point
    if all(
        all(
            not valid_condition(trial["condition"], configuration, order)
            for trial in data["trials"]
        )
        for configuration in ctx["subspace_points"]
    ):
        report(
            LogLevel.ERROR,
            "no compatible trial commands for the given subspace",
            hint="your trial conditions may be too strict, try relaxing them or adding more trials.",
        )


def run_subspace_trials(ctx):
    args = ctx["args"]
    data = ctx["data"]
    order = ctx["order"]
    subspace = ctx["subspace"]

    if args.dry_run:
        for i, configuration in enumerate(ctx["subspace_points"], start=1):
            point = {key: x for key, x in zip(order, configuration)}
            if valid_conditions(configuration, order):
                report(
                    LogLevel.INFO,
                    get_progress(i, ctx["subspace_size"]),
                    "dry run",
                    point_to_string(point),
                )
    else:
        with open(ctx["output"], "a") as f:
            for i, configuration in enumerate(ctx["subspace_points"], start=1):
                run_point_trials(ctx, f, i, configuration)
                f.flush()


def build_preset(ctx, preset_name):
    pass


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
        if valid_conditions(point, order):
            ctx["subspace_points"].append(point)
    ctx["subspace_size"] = len(ctx["subspace_points"])

    ctx["subspace_values"] = {
        key: [x["value"] for x in subspace[key]] for key in subspace
    }
    ctx["subspace_names"] = {
        key: [x["name"] for x in subspace[key]] for key in subspace
    }

    if ctx["subspace_size"] == 0:
        report(LogLevel.WARNING, "empty subspace")
    else:
        report(LogLevel.INFO, "subspace size", ctx["subspace_size"])


def validate_presets(ctx):
    available = ctx["data"]["presets"].keys()
    args = ctx["args"]
    for preset_name in args.presets:
        if preset_name not in available:
            hint = "available presets: {}".format(", ".join(available))
            report(
                LogLevel.FATAL,
                "invalid preset",
                preset_name,
                hint=hint,
            )


def validate_setup(ctx):
    setup = ctx["data"]["setup"]
    # we assume setup is normalized
    psetup = setup["point"]

    # check validity of 'on' fields
    for item in psetup:
        on = item["on"]
        if not isinstance(on, (list, type(None))):
            report(LogLevel.FATAL, "point setup 'on' must be a list or None")
        for dim in item["on"]:
            if not isinstance(dim, str):
                report(LogLevel.FATAL, "every 'on' dimension must be a string")

    # check validity of 'parallel' fields
    for item in psetup:
        parallel = item["parallel"]
        if not isinstance(parallel, (bool, list)):
            report(LogLevel.FATAL, "point setup 'parallel' must be a boolean or a list")
        if isinstance(parallel, list):
            wrong = [
                x for x in parallel if not isinstance(x, str) or x not in item["on"]
            ]
            if len(wrong) > 0:
                hint = "available dimensions: {}".format(", ".join(item["on"]))
                report(
                    LogLevel.FATAL,
                    "invalid parallel dimensions",
                    ", ".join(wrong),
                    hint=hint,
                )


def get_point_setup_context(ctx, item):
    args = ctx["args"]
    data = ctx["data"]
    order = ctx["order"]
    dims = item["on"] or data["space"].keys()

    if isinstance(item["parallel"], bool):
        if item["parallel"]:
            item["parallel"] = dims
        else:
            item["parallel"] = []
    elif isinstance(item["parallel"], list):
        for dim in item["parallel"]:
            if not isinstance(dim, str):
                report(LogLevel.FATAL, "parallel dimensions must be strings")
            if dim not in dims:
                hint = "available dimensions: {}".format(", ".join(dims))
                report(
                    LogLevel.FATAL,
                    "invalid parallel dimension",
                    dim,
                    hint=hint,
                )

    # create valid subspace for parallel setup
    subspace = ctx["subspace"]
    parallel_dims = set(item["parallel"])
    sequential_dims = set(dims) - parallel_dims
    parallel_dims = [x for x in order if x in parallel_dims]
    sequential_dims = [x for x in order if x in sequential_dims]
    parallel_space = [subspace[k] for k in parallel_dims]
    sequential_space = [subspace[k] for k in sequential_dims]

    return {
        "parallel_dims": parallel_dims,
        "sequential_dims": sequential_dims,
        "parallel_space": parallel_space,
        "sequential_space": sequential_space,
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
    os.makedirs(args.temp_dir, exist_ok=True)
    ctx["random_key"] = "".join(
        random.choices(string.ascii_letters + string.digits, k=8)
    )
    report(LogLevel.INFO, "working directory", os.getcwd())
    report(LogLevel.INFO, "input configurations", ", ".join(args.inputs))
    report(LogLevel.INFO, "output data", ctx["output"])
    report(LogLevel.INFO, "temp directory", args.temp_dir)
    report(LogLevel.INFO, "random key", ctx["random_key"])


def normalize_point_setup(psetup):
    if isinstance(psetup, str):
        psetup = [
            {"commands": [x], "on": None, "parallel": False}
            for x in normalize_command_list(psetup)
        ]
    elif isinstance(psetup, list):
        normalized = []
        for item in psetup:
            unexpected = [x for x in item if x not in ["commands", "on", "parallel"]]
            if len(unexpected) > 0:
                report(
                    LogLevel.WARNING,
                    "point setup item has unexpected fields",
                    ", ".join(unexpected),
                    hint="fields: 'commands', 'on', 'parallel'",
                )
            if isinstance(item, str):
                normalized.append({"commands": [item], "on": None, "parallel": False})
            elif isinstance(item, dict):
                if "commands" in item:
                    normalized.append(
                        {
                            "commands": normalize_command_list(item["commands"]),
                            "on": item.get("on", None),
                            "parallel": item.get("parallel", False),
                        }
                    )
                else:
                    report(
                        LogLevel.FATAL,
                        "point setup item must have 'commands' field",
                    )
            else:
                report(LogLevel.FATAL, "point setup must be a string or a list")
    elif isinstance(psetup, dict):
        report(LogLevel.FATAL, "point setup must be a string or a list")

    return normalized


def normalize_setup(setup):
    normalized = {"global": [], "point": []}

    if not isinstance(setup, dict):
        report(LogLevel.FATAL, "setup must have fields 'global' and/or 'point'")

    if "global" in setup:
        normalized["global"] = normalize_command_list(setup["global"])

    if "point" in setup:
        normalized["point"] = normalize_point_setup(setup["point"])

    return normalized


def run_experiments(ctx, preset_name=None):
    if preset_name is None:
        ctx["subspace"] = ctx["space"].copy()
    else:
        ctx["current_preset"] = preset_name
        build_preset(ctx, preset_name)
        build_subspace(ctx, preset_name)

    validate_subspace(ctx)
    validate_points(ctx)
    validate_setup(ctx)
    run_setup(ctx)
    run_subspace_trials(ctx)


def launch(args):
    ctx = {"args": args}
    validate_args(ctx)
    read_configurations(ctx)
    build_environment(ctx)
    build_space(ctx)
    define_order(ctx)
    validate_yvars(ctx)
    validate_presets(ctx)

    if len(args.presets) > 0:
        for preset_name in args.presets:
            report(LogLevel.INFO, "running preset", preset_name)
            run_experiments(ctx, preset_name)
            report(LogLevel.INFO, "completed preset", preset_name)
    else:
        run_experiments(ctx, preset_name=None)

    report(LogLevel.INFO, "finished")
    if not args.dry_run:
        y_axis = ctx["data"]["metrics"].keys()
        hint = "use `yuclid plot {} -y {}` to analyze the results".format(
            ctx["output"], ",".join(y_axis)
        )
        report(LogLevel.INFO, "output data written to", ctx["output"], hint=hint)

from yuclid.log import LogLevel, report
from datetime import datetime
import concurrent.futures
import pandas as pd
import subprocess
import itertools
import json
import re
import os


def substitute_point_yvars(x, point_map, point_id):
    # replace ${yuclid.<name>} and ${yuclid.@} with point values
    value_pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)(?:\.value)?\}"
    name_pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\.name\}"
    matches = re.findall(name_pattern, x)
    for name in matches:
        if name not in point_map:
            report(
                LogLevel.FATAL,
                f"point variable '{name}' not found in point_map",
                hint=f"available variables: {', '.join(point_map.keys())}",
            )
    y = re.sub(value_pattern, lambda m: str(point_map[m.group(1)]["value"]), x)
    y = re.sub(name_pattern, lambda m: str(point_map[m.group(1)]["name"]), y)
    if point_id is not None:
        value_pattern = r"\$\{yuclid\.\@\}"
        y = re.sub(value_pattern, lambda m: f"{point_id}", y)
    return y


def substitute_global_yvars(x, subspace):
    # replace ${yuclid.<name>.values} and ${yuclid.<name>.names}
    subspace_values = {k: [str(x["value"]) for x in v] for k, v in subspace.items()}
    subspace_names = {k: {x["name"] for x in v} for k, v in subspace.items()}
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\.values\}"
    y = re.sub(pattern, lambda m: " ".join(subspace_values[m.group(1)]), x)
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\.names\}"
    y = re.sub(pattern, lambda m: " ".join(subspace_names[m.group(1)]), y)
    return y


def validate_point_yvars(space, exps):
    for exp in exps:
        matches = re.findall(
            r"\$\{yuclid\.([a-zA-Z0-9_@]+)(?:\.(?:name|value|names|values))?\}", exp
        )
        for dim in matches:
            if dim not in space and dim != "@":
                report(LogLevel.FATAL, f"invalid variable 'yuclid.{dim}'", exp)


def validate_global_yvars(space, exps):
    for exp in exps:
        exp = str(exp)
        point_matches = re.findall(
            r"\$\{yuclid\.([a-zA-Z0-9_@]+)(?:\.(?:name|value))?\}", exp
        )
        for dim in point_matches:
            hint = (
                "maybe you meant ${{yuclid.{}.names}} or ${{yuclid.{}.values}}?".format(
                    dim, dim
                )
            )
            report(
                LogLevel.FATAL,
                "wrong use of yuclid point variable 'yuclid.{}'".format(dim),
                hint=hint,
            )
        global_matches = re.findall(
            r"\$\{yuclid\.([a-zA-Z0-9_@]+)\.(?:names|values)\}", exp
        )
        for dim in global_matches:
            if dim not in space:
                report(LogLevel.FATAL, f"invalid variable 'yuclid.{dim}'", exp)


def validate_yvars_in_env(space, env):
    validate_global_yvars(space, env.values())


def validate_yvars_in_setup(space, setup):
    validate_global_yvars(space, setup["global"])

    for point in setup["point"]:
        validate_point_yvars(space, point["commands"])


def validate_yvars_in_trials(space, trials):
    commands = [trial["command"] for trial in trials]
    validate_point_yvars(space, commands)


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


def aggregate_input_data(settings):
    data = {
        "env": {},
        "setup": {"global": [], "point": []},
        "space": {},
        "trials": [],
        "metrics": [],
        "presets": {},
        "order": [],
    }

    for file in settings["inputs"]:
        with open(file, "r") as f:
            current = normalize_data(load_json(f))
            for key, val in current.items():
                if key in ["env", "space", "presets"]:
                    data[key].update(val)
                elif key in ["trials", "metrics", "order"]:
                    data[key].extend(val)
                elif key == "setup":
                    for subkey, subval in val.items():
                        if data[key].get(subkey) is None:
                            # undefined dimensions are overridden
                            data[key][subkey] = subval
                        else:
                            data[key].setdefault(subkey, []).extend(subval)
                elif key == "setup":
                    data[key]["setup"]["global"] += val["setup"]["global"]
                    data[key]["setup"]["point"] += val["setup"]["point"]

    data["order"] = remove_duplicates(data["order"])

    if len(data["trials"]) == 0:
        report(LogLevel.FATAL, "no valid trials found")

    if len(data["metrics"]) == 0:
        report(LogLevel.WARNING, "no metrics found. Trials will not be evaluated")

    return data


def remove_duplicates(items):
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def build_environment(settings, data):
    if settings["dry_run"]:
        for key, value in data["env"].items():
            report(LogLevel.INFO, "dry env", f'{key}="{value}"')
        env = dict()
    else:
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
        env = resolved_env
    return env


def apply_user_selectors(settings, subspace):
    all_selectors = dict(pair.split("=") for pair in settings["select"])
    for dim, csv_selection in all_selectors.items():
        selectors = csv_selection.split(",")
        if dim not in subspace:
            report(
                LogLevel.FATAL,
                "unknown dimension in selector",
                dim,
                hint="available dimensions: {}".format(", ".join(subspace.keys())),
            )
        if subspace[dim] is None:
            selection = [normalize_point(x) for x in selectors]
        else:
            selection = []
            valid = {str(x["name"]) for x in subspace[dim]}
            for selector in selectors:
                if selector in valid:
                    for x in subspace[dim]:
                        if x["name"] == selector:
                            selection.append(x)
                else:
                    report(
                        LogLevel.ERROR,
                        "invalid selector",
                        selector,
                        hint="available: {}".format(", ".join(valid)),
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


def normalize_metrics(metrics):
    valid = {"name", "command", "condition"}
    normalized = []
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                report(LogLevel.FATAL, "each metric must be a dict", metric)
            if "name" not in metric:
                report(LogLevel.FATAL, "each metric must have a 'name' field", metric)
            if "command" not in metric:
                report(
                    LogLevel.FATAL, "each metric must have a 'command' field", metric
                )
            if not set(metric.keys()).issubset(valid):
                report(
                    LogLevel.WARNING,
                    "metric has unexpected fields",
                    ", ".join(set(metric.keys()) - valid),
                    hint="valid fields: {}".format(", ".join(valid)),
                )
            normalized.append(
                {
                    "name": metric["name"],
                    "command": normalize_command(metric["command"]),
                    "condition": metric.get("condition", "True"),
                }
            )
    elif isinstance(metrics, dict):
        for name, command in metrics.items():
            if not isinstance(command, str):
                report(LogLevel.FATAL, "metric command must be a string", command)
            normalized.append(
                {
                    "name": name,
                    "command": normalize_command(command),
                    "condition": "True",
                }
            )
    return normalized


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
    valid_fields = {"name", "value", "condition", "setup"}
    if isinstance(x, (str, int, float)):
        normalized = {"name": str(x), "value": x, "condition": "True", "setup": []}
    elif isinstance(x, dict):
        if not set(x.keys()).issubset(valid_fields):
            report(
                LogLevel.WARNING,
                "point has unexpected fields",
                ", ".join(set(x.keys()) - valid_fields),
                hint="valid fields: {}".format(", ".join(valid_fields)),
            )
        if "value" in x:
            normalized = {
                "name": str(x.get("name", x["value"])),
                "value": x["value"],
                "condition": normalize_condition(x.get("condition", "True")),
                "setup": normalize_command_list(x.get("setup", [])),
            }
        else:
            report(LogLevel.FATAL, "points must have a 'value' field", x)
    else:
        report(LogLevel.FATAL, "point must be a string, int, float or a dict", x)
    return normalized


def normalize_trials(trial):
    valid = {"command", "condition", "metrics"}
    if isinstance(trial, str):
        return [{"command": trial, "condition": "True", "metrics": None}]
    elif isinstance(trial, list):
        items = []
        for item in trial:
            normalized_item = {"command": None, "condition": "True", "metrics": None}
            if isinstance(item, str):
                normalized_item["command"] = normalize_command(item)
            elif isinstance(item, dict):
                # check for invalid fields
                invalid_fields = set(item.keys()) - valid
                if len(invalid_fields) > 0:
                    report(
                        LogLevel.WARNING,
                        "trial item has unexpected fields",
                        ", ".join(invalid_fields),
                        hint="valid fields: {}".format(", ".join(valid)),
                    )
                if "command" not in item:
                    report(
                        LogLevel.FATAL, "each trial item must have a 'command' field"
                    )
                    return None
                normalized_item["command"] = normalize_command(item["command"])
                normalized_item["condition"] = item.get("condition", "True")
                normalized_item["metrics"] = item.get("metrics", None)
            items.append(normalized_item)
        return items
    else:
        report(LogLevel.FATAL, "trial must be a string or a list of strings")
        return None


def normalize_space_values(space):
    normalized = dict()
    for key, values in space.items():
        if key.endswith(":py"):
            name = key.split(":")[-2]
            if not isinstance(values, str):
                report(LogLevel.FATAL, "python command must be a string", key)
            result = eval(values)
            if not isinstance(result, list):
                report(
                    LogLevel.FATAL, "python command generated non-list values", values
                )
            normalized[name] = [normalize_point(x) for x in result]
        elif values is not None:
            normalized[key] = []
            for x in values:
                normalized[key].append(normalize_point(x))
        else:
            normalized[key] = None
    return normalized


def normalize_data(json_data):
    valid_fields = {
        "env",
        "setup",
        "space",
        "trials",
        "metrics",
        "presets",
        "order",
    }

    normalized = dict()
    for key in json_data.keys():
        if key in valid_fields:
            normalized[key] = json_data[key]
        else:
            report(
                LogLevel.WARNING,
                "unknown field in configuration",
                key,
                hint="available fields: {}".format(", ".join(valid_fields)),
            )

    space = normalize_space_values(json_data.get("space", {}))

    normalized["space"] = space
    normalized["trials"] = normalize_trials(json_data.get("trials", []))
    normalized["setup"] = normalize_setup(json_data.get("setup", {}), space)
    normalized["metrics"] = normalize_metrics(json_data.get("metrics", []))
    normalized["presets"] = json_data.get("presets", dict())

    for trial in normalized["trials"]:
        if trial["metrics"] is not None:
            metric_names = [m["name"] for m in normalized["metrics"]]
            if not all(m in metric_names for m in trial["metrics"]):
                report(
                    LogLevel.FATAL,
                    "trial references unknown metrics",
                    ", ".join(trial["metrics"]),
                    hint="available metrics: {}".format(
                        ", ".join({m["name"] for m in normalized["metrics"]})
                    ),
                )

    return normalized


def reorder_dimensions(dimensions, order):
    reordered = list(dimensions)
    for k in order:
        if k in reordered:
            reordered.append(reordered.pop(reordered.index(k)))
    return reordered


def define_order(settings, data):
    space = data["space"]

    available = space.keys()
    invalid_order_keys = [
        k for k in set(settings["order"] + data["order"]) if k not in available
    ]
    if len(invalid_order_keys) > 0:
        hint = "available values: {}".format(", ".join(available))
        report(
            LogLevel.FATAL,
            "invalid order values",
            ", ".join(invalid_order_keys),
            hint=hint,
        )

    dims = list(space.keys())
    final_order = reorder_dimensions(dims, data["order"])
    final_order = reorder_dimensions(final_order, settings["order"])

    return final_order


def apply_preset(data, preset_name):
    space = data["space"]
    space_names = {
        dim: [x["name"] for x in space[dim]] for dim in space if space[dim] is not None
    }
    preset_space = data["presets"][preset_name]
    subspace = space.copy()

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
                # definition via name
                new_items.append(next(x for x in space[dim] if x["name"] == str(item)))

        if len(wrong) > 0:
            hint = "available names: {}".format(", ".join(space_names[dim]))
            report(
                LogLevel.FATAL,
                f"unknown name in preset '{preset_name}'",
                ", ".join(wrong),
                hint=hint,
            )
        subspace[dim] = new_items

    for dim, items in subspace.items():
        if items is not None and len(items) == 0:
            report(LogLevel.ERROR, f"empty dimension in preset '{preset_name}'", dim)

    return subspace


def run_point_command(command, execution, point, on_dims):
    gcommand = substitute_global_yvars(command, execution["subspace"])
    on_dims_0 = [dim.split(".")[0] for dim in on_dims]
    suborder = [d for d in execution["order"] if d in on_dims_0]
    point_map = {key: x for key, x in zip(suborder, point)}
    pcommand = substitute_point_yvars(gcommand, point_map, None)

    if not valid_conditions(point, suborder):
        return

    if execution["dry_run"]:
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
            env=execution["env"],
        )
        if result.returncode != 0:
            report(
                LogLevel.ERROR,
                "point setup",
                f"'{command}'",
                f"returned {result.returncode}",
            )


def run_points_sequential(command, item_plan, execution, on_dims, par_config):
    sequential_space = item_plan["sequential_space"]
    parallel_dims = item_plan["parallel_dims"]
    seq_points = list(itertools.product(*sequential_space))
    sequential_dims = item_plan["sequential_dims"]
    order = execution["order"]

    named_par_config = [(name, x) for name, x in zip(parallel_dims, par_config)]
    if len(sequential_dims) == 0:
        final_config = [x[1] for x in named_par_config]
        run_point_command(command, execution, final_config, on_dims)
        return

    for seq_config in seq_points:
        named_seq_config = [(dim, x) for dim, x in zip(sequential_dims, seq_config)]
        named_ordered_config = sorted(
            named_par_config + named_seq_config, key=lambda x: order.index(x[0])
        )
        final_config = [x[1] for x in named_ordered_config]
        run_point_command(command, execution, final_config, on_dims)


def run_point_setup_item(item, execution):
    commands = item["commands"]
    on_dims = item["on"]
    order = execution["order"]
    subspace = execution["subspace"]
    item_plan = get_point_item_plan(item, subspace, order)

    parallel_space = item_plan["parallel_space"]
    parallel_dims = item_plan["parallel_dims"]

    num_parallel_dims = len(parallel_space)
    if num_parallel_dims == 0:
        max_workers = 1
    else:
        max_workers = min(execution["subspace_size"], os.cpu_count() or 1)
    report(LogLevel.INFO, f"using {max_workers} workers for point setup")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        par_points = list(itertools.product(*parallel_space))

        for i, command in enumerate(commands, start=1):
            if len(parallel_dims) == 0:
                run_points_sequential(command, item_plan, execution, on_dims, [])
            else:
                futures = []
                for j, par_config in enumerate(par_points, start=1):
                    future = executor.submit(
                        run_points_sequential,
                        command,
                        item_plan,
                        execution,
                        on_dims,
                        par_config,
                    )
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    exc = future.exception()
                    if exc is not None:
                        report(LogLevel.ERROR, "point setup failed", command)


def run_point_setup(data, execution):
    for item in data["setup"]["point"]:
        if item["on"] is None:
            item["on"] = execution["subspace"].keys()

        # reordering
        item["on"] = [x for x in item["on"] if x.split(".")[0] in execution["order"]]
        if len(item["on"]) == 0:
            report(
                LogLevel.WARNING,
                "point setup item has no valid 'on' dimensions. Skipping",
                item,
            )
        else:
            if execution["dry_run"]:
                report(LogLevel.INFO, "starting dry point setup")
            else:
                report(LogLevel.INFO, "starting point setup")

            run_point_setup_item(item, execution)

            if execution["dry_run"]:
                report(LogLevel.INFO, "dry point setup completed")
            else:
                report(LogLevel.INFO, "point setup completed")


def run_global_setup(data, execution):
    subspace = execution["subspace"]
    setup_commands = data["setup"]["global"]
    # gather setup commands from space
    for key, values in subspace.items():
        for value in values:
            if len(value["setup"]) > 0:
                for cmd in value["setup"]:
                    setup_commands.append(cmd)

    if execution["dry_run"]:
        report(LogLevel.INFO, "starting dry global setup")
    else:
        report(LogLevel.INFO, "starting global setup")

    errors = False
    for command in setup_commands:
        if execution["dry_run"]:
            report(LogLevel.INFO, "dry run", command)
        else:
            command = substitute_global_yvars(command, subspace)
            result = subprocess.run(
                command,
                shell=True,
                universal_newlines=True,
                capture_output=False,
                env=execution["env"],
            )
            if result.returncode != 0:
                errors = True
                report(
                    LogLevel.ERROR,
                    "setup",
                    f"'{command}'",
                    f"returned {result.returncode}",
                )
    if errors:
        report(LogLevel.WARNING, "errors have occurred during setup")
        report(LogLevel.INFO, "setup failed")
    if execution["dry_run"]:
        report(LogLevel.INFO, "dry setup completed")
    else:
        report(LogLevel.INFO, "setup completed")


def run_setup(settings, data, execution):
    run_global_setup(data, execution)
    run_point_setup(data, execution)


def point_to_string(point):
    return ".".join([str(x["name"]) for x in point])


def metrics_to_string(metric_values):
    return " ".join([f"{m}={v}" for m, v in metric_values.items()])


def get_progress(i, subspace_size):
    return "[{}/{}]".format(i, subspace_size)


def run_point_trials(settings, data, execution, f, i, point):
    os.makedirs(
        os.path.join(settings["temp_dir"], settings["now"]),
        exist_ok=True,
    )

    point_map = {key: x for key, x in zip(execution["order"], point)}
    report(
        LogLevel.INFO,
        get_progress(i, execution["subspace_size"]),
        point_to_string(point),
        "started",
    )

    compatible_trials, compatible_metrics = get_compatible_trials_and_metrics(
        data, point, execution
    )

    if len(compatible_metrics) == 0:
        report(LogLevel.WARNING, point_to_string(point), "no compatible metrics found")

    if len(compatible_trials) == 0 or len(compatible_metrics) == 0:
        report(
            LogLevel.WARNING,
            point_to_string(point),
            "no compatible trials found",
            hint="try relaxing your trial conditions or adding more trials.",
        )

    i_padded = str(i).zfill(len(str(execution["subspace_size"])))

    for j, trial in enumerate(compatible_trials):
        point_id = os.path.join(
            settings["temp_dir"],
            settings["now"],
            f"{i_padded}." + point_to_string(point) + f"_trial{j}",
        )

        command = substitute_global_yvars(trial["command"], execution["subspace"])
        command = substitute_point_yvars(command, point_map, point_id)
        command_output = subprocess.run(
            command,
            shell=True,
            env=execution["env"],
            universal_newlines=True,
            capture_output=True,
        )

        with open(f"{point_id}.out", "w") as output_file:
            if command_output.stdout:
                output_file.write(command_output.stdout)

        with open(f"{point_id}.err", "w") as error_file:
            if command_output.stderr:
                error_file.write(command_output.stderr)

        if command_output.returncode != 0:
            hint = "check the following files for more details:\n"
            hint += f"{point_id}.out\n{point_id}.err"
            report(
                LogLevel.ERROR,
                point_to_string(point),
                f"failed trial command '{command}' (code {command_output.returncode})",
                hint=hint,
            )

    collected_metrics = dict()
    for metric in compatible_metrics:
        command = substitute_global_yvars(metric["command"], execution["subspace"])
        command = substitute_point_yvars(command, point_map, point_id)
        command_output = subprocess.run(
            command,
            shell=True,
            universal_newlines=True,
            capture_output=True,
            env=execution["env"],
        )

        def complain():
            hint = "check the following files for more details:\n"
            hint += f"{point_id}.out\n{point_id}.err\n"
            report(
                LogLevel.ERROR,
                point_to_string(point),
                "failed metric '{}' (code {})".format(
                    metric["name"], command_output.returncode
                ),
                hint=hint,
            )

        if command_output.returncode != 0:
            complain()
        else:
            output_lines = command_output.stdout.strip().split("\n")

            def int_or_float(x):
                try:
                    return int(x)
                except ValueError:
                    try:
                        return float(x)
                    except ValueError:
                        complain()

            collected_metrics[metric["name"]] = [
                int_or_float(line) for line in output_lines
            ]

    metric_values_df = pd.DataFrame.from_dict(
        collected_metrics, orient="index"
    ).transpose()
    if not settings["fold"]:
        NaNs = metric_values_df.columns[metric_values_df.isnull().any()]
        if len(NaNs) > 0:
            report(
                LogLevel.WARNING,
                "the following metrics generated some NaNs",
                " ".join(list(NaNs)),
            )

    result = {k: x["name"] for k, x in point_map.items()}
    if settings["fold"]:
        result.update(metric_values_df.to_dict(orient="list"))
        f.write(json.dumps(result) + "\n")
    else:
        for record in metric_values_df.to_dict(orient="records"):
            result.update(record)
            f.write(json.dumps(result) + "\n")

    report(LogLevel.INFO, "obtained", metrics_to_string(collected_metrics))
    for metric_name, values in collected_metrics.items():
        if len(values) > 1:
            report(
                LogLevel.INFO,
                "median",
                metric_name,
                f"{pd.Series(values).median():.3f}",
            )
    report(
        LogLevel.INFO,
        get_progress(i, execution["subspace_size"]),
        point_to_string(point),
        "completed",
    )
    f.flush()


def valid_conditions(point, order):
    point_context = {}
    yuclid = {name: x["value"] for name, x in zip(order, point)}
    point_context["yuclid"] = type("Yuclid", (), yuclid)()
    conditions = [x["condition"] for x in point if "condition" in x]
    return all(eval(c, point_context) for c in conditions)


def valid_condition(condition, point, order):
    point_context = {}
    yuclid = {name: x["value"] for name, x in zip(order, point)}
    point_context["yuclid"] = type("Yuclid", (), yuclid)()
    return eval(condition, point_context)


def validate_execution(execution, data):
    # checking if there's at least of compatible trial command for each point
    for point in execution["subspace_points"]:
        compatible_trials, compatible_metrics = get_compatible_trials_and_metrics(
            data, point, execution
        )
        if len(compatible_trials) == 0:
            report(
                LogLevel.ERROR,
                "no compatible trials found for point",
                point_to_string(point),
                hint="try relaxing your trial conditions or adding more trials.",
            )
        if len(execution["metrics"] or []) > 0:
            compatible_metric_names = {m["name"] for m in compatible_metrics}
            incompatible = [
                m for m in execution["metrics"] if m not in compatible_metric_names
            ]
            if len(incompatible) > 0:
                report(
                    LogLevel.ERROR,
                    "some metrics are not compatible with {}".format(
                        point_to_string(point)
                    ),
                    ", ".join(incompatible),
                    hint="try relaxing your metric conditions or adding more metrics.",
                )


def get_compatible_trials_and_metrics(data, point, execution):
    all_metric_names = {m["name"] for m in data["metrics"]}
    selected_metric_names = execution["metrics"] or all_metric_names
    valid_metrics = [
        metric
        for metric in data["metrics"]
        if metric["name"] in selected_metric_names
        and valid_condition(metric["condition"], point, execution["order"])
    ]
    valid_metric_names = {m["name"] for m in valid_metrics}
    compatible_trials = [
        trial
        for trial in data["trials"]
        if valid_condition(trial["condition"], point, execution["order"])
        and (
            trial["metrics"] is None
            or any(m in valid_metric_names for m in trial["metrics"])
        )
    ]
    compatible_metrics = [
        metric
        for metric in valid_metrics
        if any(
            trial["metrics"] is None or metric["name"] in trial["metrics"]
            for trial in compatible_trials
        )
    ]
    return compatible_trials, compatible_metrics


def run_subspace_trials(settings, data, execution):
    if settings["dry_run"]:
        for i, point in enumerate(execution["subspace_points"], start=1):
            point_map = {key: x for key, x in zip(execution["order"], point)}
            if valid_conditions(point, execution["order"]):
                compatible_trials, compatible_metrics = (
                    get_compatible_trials_and_metrics(data, point, execution)
                )
                report(
                    LogLevel.INFO,
                    get_progress(i, execution["subspace_size"]),
                    "dry run",
                    point_to_string(point),
                )
    else:
        output_dir = os.path.dirname(settings["output"])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(settings["output"], "a") as f:
            for i, point in enumerate(execution["subspace_points"], start=1):
                run_point_trials(settings, data, execution, f, i, point)
                f.flush()


def validate_dimensions(subspace):
    undefined = [k for k, v in subspace.items() if v is None]
    if len(undefined) > 0:
        hint = "define dimensions with presets or select them with --select. "
        hint += "E.g. --select {}=value1,value2".format(undefined[0])
        report(LogLevel.FATAL, "dimensions undefined", ", ".join(undefined), hint=hint)


def prepare_subspace_execution(subspace, order, env, metrics, dry_run):
    ordered_subspace = [subspace[x] for x in order]

    execution = dict()
    execution["subspace_points"] = []
    for point in itertools.product(*ordered_subspace):
        if valid_conditions(point, order):
            execution["subspace_points"].append(point)
    execution["subspace_size"] = len(execution["subspace_points"])

    execution["subspace_values"] = {
        key: [x["value"] for x in subspace[key]] for key in subspace
    }
    execution["subspace_names"] = {
        key: [x["name"] for x in subspace[key]] for key in subspace
    }
    execution["subspace"] = subspace
    execution["order"] = order
    execution["env"] = env
    execution["dry_run"] = dry_run
    execution["metrics"] = metrics

    if execution["subspace_size"] == 0:
        report(LogLevel.WARNING, "empty subspace")
    else:
        report(LogLevel.INFO, "subspace size", execution["subspace_size"])
    return execution


def validate_presets(settings, data):
    available = data["presets"].keys()
    for preset_name in settings["presets"]:
        if preset_name not in available:
            hint = "available presets: {}".format(", ".join(available))
            report(
                LogLevel.FATAL,
                "invalid preset",
                preset_name,
                hint=hint,
            )


def get_point_item_plan(item, subspace, order):
    all_dims = item["on"] or [f"{k}.values" for k in subspace.keys()]

    name_dims = [d.split(".")[0] for d in all_dims if d.endswith(".names")]
    value_dims = [d.split(".")[0] for d in all_dims if d.endswith(".values")]

    if isinstance(item["parallel"], bool):
        if item["parallel"]:
            item["parallel"] = item["on"]
        else:
            item["parallel"] = []

    # collect parallel and sequential dimensions
    parallel_dims = set(item["parallel"])
    sequential_dims = set([x.split(".")[0] for x in all_dims]) - parallel_dims

    # reordering
    parallel_dims = reorder_dimensions(parallel_dims, order)
    sequential_dims = reorder_dimensions(sequential_dims, order)

    # building the plan_subspace which is just the subspace where `name_dims`
    # have items with no conditions and name==value

    plan_subspace = dict()
    plan_subspace.update({k: subspace[k] for k in value_dims})

    subspace_names = {k: {x["name"] for x in subspace[k]} for k in name_dims}
    plan_subspace.update(
        {d: [normalize_point(x) for x in names] for d, names in subspace_names.items()}
    )

    # building space
    parallel_space = [plan_subspace[k] for k in parallel_dims]
    sequential_space = [plan_subspace[k] for k in sequential_dims]

    # name_dims don't need to have conditions, and their name is their value
    return {
        "parallel_dims": parallel_dims,
        "sequential_dims": sequential_dims,
        "parallel_space": parallel_space,
        "sequential_space": sequential_space,
    }


def build_settings(args):
    settings = dict(vars(args))

    # inputs
    settings["inputs"] = []
    for file in args.inputs:
        if not os.path.isfile(file):
            report(LogLevel.ERROR, f"'{file}' does not exist")
        else:
            settings["inputs"].append(file)
    os.makedirs(args.temp_dir, exist_ok=True)

    # output
    settings["now"] = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    filename = "results.{}.json".format(settings["now"])
    if args.output is None and args.output_dir is None:
        settings["output"] = filename
    elif args.output is not None and args.output_dir is not None:
        report(LogLevel.FATAL, "either --output or --output-dir must be specified")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        settings["output"] = os.path.join(args.output_dir, filename)
    else:
        settings["output"] = args.output

    report(LogLevel.INFO, "working directory", os.getcwd())
    report(LogLevel.INFO, "input configurations", ", ".join(args.inputs))
    report(LogLevel.INFO, "output data", settings["output"])
    report(
        LogLevel.INFO,
        "temp directory",
        os.path.join(settings["temp_dir"], settings["now"]),
    )
    return settings


def normalize_point_setup(point_setup, space):
    if isinstance(point_setup, str):
        point_setup = [
            {"commands": [x], "on": None, "parallel": False}
            for x in normalize_command_list(point_setup)
        ]
    elif isinstance(point_setup, list):
        normalized_items = []
        for item in point_setup:
            unexpected = [x for x in item if x not in ["commands", "on", "parallel"]]
            if len(unexpected) > 0:
                report(
                    LogLevel.WARNING,
                    "point setup item has unexpected fields",
                    ", ".join(unexpected),
                    hint="fields: 'commands', 'on', 'parallel'",
                )
            if isinstance(item, str):
                normalized_items.append(
                    {"commands": [item], "on": None, "parallel": False}
                )
            elif isinstance(item, dict):
                if "commands" in item:
                    normalized_item = {
                        "commands": normalize_command_list(item["commands"]),
                        "on": item.get("on", None),
                        "parallel": item.get("parallel", False),
                    }
                    if normalized_item["on"] is None:
                        normalized_item["on"] = list(space.keys())

                    # adding .values to 'on' dimensions
                    normalized_item["on"] = [
                        x if "." in x else x + ".values" for x in normalized_item["on"]
                    ]

                    if normalized_item["parallel"] == True:
                        normalized_item["parallel"] = [
                            x.split(".")[0] for x in normalized_item["on"]
                        ]
                    normalized_items.append(normalized_item)
                else:
                    report(
                        LogLevel.FATAL,
                        "point setup item must have 'commands' field",
                    )
            else:
                report(LogLevel.FATAL, "point setup must be a string or a list")
    elif isinstance(point_setup, dict):
        report(LogLevel.FATAL, "point setup must be a string or a list")

    # check validity of 'on' fields
    for item in normalized_items:
        if not isinstance(item["on"], (list, type(None))):
            report(LogLevel.FATAL, "point setup 'on' must be a list or None")
        for dim in item["on"]:
            if not isinstance(dim, str):
                report(LogLevel.FATAL, "every 'on' dimension must be a string")
            elif dim.split(".")[0] not in space:
                available = list(space.keys())
                hint = "available dimensions: {}".format(", ".join(available))
                report(
                    LogLevel.FATAL,
                    "point setup 'on' dimension not in space",
                    dim,
                    hint=hint,
                )

    # check validity of 'parallel' fields
    for item in normalized_items:
        parallel = item["parallel"]
        if not isinstance(parallel, (bool, list)) or (
            isinstance(parallel, list) and any(not isinstance(x, str) for x in parallel)
        ):
            report(
                LogLevel.FATAL,
                "point setup 'parallel' must be a boolean or a list of strings",
            )
        if isinstance(parallel, list):
            for x in parallel:
                if x not in space:
                    report(
                        LogLevel.FATAL,
                        "point setup 'parallel' dimension not in space",
                        x,
                        hint="available dimensions: {}".format(", ".join(space.keys())),
                    )
            wrong = [x for x in parallel if x + ".values" not in item["on"]]
            if len(wrong) > 0:
                hint = "available dimensions: {}".format(", ".join(item["on"]))
                report(
                    LogLevel.FATAL,
                    "invalid parallel dimensions",
                    ", ".join(wrong),
                    hint=hint,
                )

    return normalized_items


def normalize_setup(setup, space):
    normalized = {"global": [], "point": []}

    if not isinstance(setup, dict):
        report(LogLevel.FATAL, "setup must have fields 'global' and/or 'point'")

    if "global" in setup:
        normalized["global"] = normalize_command_list(setup["global"])

    if "point" in setup:
        normalized["point"] = normalize_point_setup(setup["point"], space)

    return normalized


def print_subspace(subspace):
    for dim, values in subspace.items():
        names = remove_duplicates([x["name"] for x in values])
        report(LogLevel.INFO, "subspace.{}: {}".format(dim, ", ".join(names)))


def run_experiments(settings, data, order, env, preset_name=None):
    if preset_name is None:
        subspace = data["space"].copy()
    else:
        subspace = apply_preset(data, preset_name)

    subspace = apply_user_selectors(settings, subspace)
    validate_dimensions(subspace)
    print_subspace(subspace)
    execution = prepare_subspace_execution(
        subspace, order, env, metrics=settings["metrics"], dry_run=settings["dry_run"]
    )
    validate_execution(execution, data)
    run_setup(settings, data, execution)
    run_subspace_trials(settings, data, execution)


def validate_settings(data, settings):
    if settings["metrics"]:
        valid = {x["name"] for x in data["metrics"]}
        wrong = [m for m in settings["metrics"] if m not in valid]
        if len(wrong) > 0:
            hint = "available metrics: {}".format(", ".join(valid))
            report(
                LogLevel.FATAL,
                "invalid metrics",
                ", ".join(wrong),
                hint=hint,
            )


def launch(args):
    settings = build_settings(args)
    data = aggregate_input_data(settings)
    validate_settings(data, settings)
    env = build_environment(settings, data)
    order = define_order(settings, data)
    validate_yvars_in_env(data["space"], data["env"])
    validate_yvars_in_setup(data["space"], data["setup"])
    validate_yvars_in_trials(data["space"], data["trials"])

    validate_presets(settings, data)

    if len(settings["presets"]) > 0:
        for preset_name in settings["presets"]:
            report(LogLevel.INFO, "running preset", preset_name)
            run_experiments(settings, data, order, env, preset_name)
            report(LogLevel.INFO, "completed preset", preset_name)
    else:
        run_experiments(settings, data, order, env, preset_name=None)

    report(LogLevel.INFO, "finished")

    if not settings["dry_run"]:
        metric_names = {m["name"] for m in data["metrics"]}
        hint = "use `yuclid plot {} -y {}` to analyze the results".format(
            settings["output"], ",".join(metric_names)
        )
        report(LogLevel.INFO, "output data written to", settings["output"], hint=hint)

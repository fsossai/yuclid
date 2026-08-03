from yuclid.log import LogLevel, report
from yuclid.plan import Plan
from datetime import datetime
import concurrent.futures
import yuclid.workspace as workspace
import yuclid.control as control
import threading
import pandas as pd
import subprocess
import itertools
import json
import csv
import signal
import sys
import time
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


def validate_point_yvars(space, exp):
    matches = re.findall(
        r"\$\{yuclid\.([a-zA-Z0-9_@]+)(?:\.(?:name|value|names|values))?\}", exp
    )
    for dim in matches:
        if dim not in space and dim != "@":
            report(LogLevel.FATAL, f"invalid variable 'yuclid.{dim}'", exp)


def validate_global_yvars(space, exp):
    exp = str(exp)
    point_matches = re.findall(
        r"\$\{yuclid\.([a-zA-Z0-9_@]+)(?:\.(?:name|value))?\}", exp
    )
    for dim in point_matches:
        hint = "maybe you meant ${{yuclid.{}.names}} or ${{yuclid.{}.values}}?".format(
            dim, dim
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
    for v in env.values():
        validate_global_yvars(space, v)


def validate_yvars_in_setup(space, setup):
    for command in setup["global"]:
        validate_global_yvars(space, command)

    for point in setup["point"]:
        validate_point_yvars(space, point["command"])


def validate_yvars_in_trials(space, trials):
    for trial in trials:
        command = validate_point_yvars(space, trial["command"])


DEFAULT_INPUTS = ["yuclid.json", "yuclid.yaml", "yuclid.yml"]


def record_columns(data, settings, order):
    """Every column a record may hold: the dimensions, then the metrics.

    CSV needs one fixed header for the whole dataset, while conditions make
    the metrics collected at a point vary, so the columns are decided from the
    configuration rather than from the first record.
    """
    selected = settings["metrics"]
    names = remove_duplicates(
        m["name"]
        for m in data["metrics"]
        if selected is None or m["name"] in selected
    )
    return list(order) + names


class RecordWriter:
    """Appends records to the result dataset, as JSON Lines or as CSV."""

    def __init__(self, stream, fmt, columns, write_header):
        self.stream = stream
        self.format = fmt
        self.columns = columns
        if fmt == "csv":
            self.writer = csv.DictWriter(
                stream,
                fieldnames=columns,
                extrasaction="ignore",
                restval="",
                # csv defaults to CRLF; stay with the line ending of every
                # other file yuclid writes, including the compiled script's
                lineterminator="\n",
            )
            if write_header:
                self.writer.writeheader()

    def write(self, record):
        if self.format == "csv":
            self.writer.writerow(record)
        else:
            self.stream.write(json.dumps(record) + "\n")

    def flush(self):
        self.stream.flush()


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


def load_yaml(f):
    try:
        import yaml
    except ImportError:
        report(
            LogLevel.FATAL,
            "reading a YAML configuration requires PyYAML",
            f.name,
            hint="install it with `pip install pyyaml`, or write the "
            "configuration in JSON",
        )
    try:
        return yaml.safe_load(f)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = ""
        if mark is not None:
            where = "at line {}, column {}: ".format(mark.line + 1, mark.column + 1)
        report(
            LogLevel.FATAL,
            "failed to parse YAML",
            f.name,
            where + str(getattr(e, "problem", None) or e),
        )


def load_config(path):
    """Read one configuration document. JSON and YAML are interchangeable."""
    with open(path, "r") as f:
        if path.lower().endswith((".yaml", ".yml")):
            data = load_yaml(f)
        else:
            data = load_json(f)
    if data is None:
        # an empty document contributes nothing
        return dict()
    if not isinstance(data, dict):
        report(
            LogLevel.FATAL,
            "a configuration must be a mapping of sections",
            path,
            hint="available sections: env, setup, space, trials, metrics, "
            "presets, order",
        )
    return data


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
        current = normalize_data(load_config(file))
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


def parse_coordinates(pairs, what="selector"):
    """`dim=v1,v2 dim2=v3` as {dim: [values]}, the way every command spells it."""
    coordinates = dict()
    for pair in pairs:
        if "=" not in pair:
            report(
                LogLevel.FATAL,
                "malformed {}".format(what),
                pair,
                hint="write it as dim=value or dim=value1,value2",
            )
        dim, values = pair.split("=", 1)
        coordinates.setdefault(dim, []).extend(values.split(","))
    return coordinates


def apply_user_selectors(settings, subspace):
    all_selectors = parse_coordinates(settings["select"])
    for dim, selectors in all_selectors.items():
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
                    # one bad name out of several is not yet the whole story:
                    # what matters is whether anything was selected at all,
                    # which is decided below
                    report(
                        LogLevel.ERROR,
                        "invalid selector",
                        selector,
                        hint="available: {}".format(", ".join(valid)),
                        fatal=False,
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
                    # like a selector, a preset defines an undefined dimension:
                    # the entry becomes both the name and the value
                    new_items.append(normalize_point(item))
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
                # every matching value, so duplicated names keep their conditions
                new_items += [x for x in space[dim] if regex.match(x["name"])]
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

    if not valid_subpoint_conditions(point, suborder):
        return

    if execution["script"] is not None:
        execution["script"].command(pcommand)
    elif execution["dry_run"]:
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
                f"'{pcommand}'",
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
    command = item["command"]
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
            if execution["script"] is not None:
                execution["script"].section("point setup")
                execution["script"].progress("point setup")
                report(LogLevel.INFO, "compiling point setup")
            elif execution["dry_run"]:
                report(LogLevel.INFO, "starting dry point setup")
            else:
                report(LogLevel.INFO, "starting point setup")

            run_point_setup_item(item, execution)

            if execution["script"] is not None:
                pass
            elif execution["dry_run"]:
                report(LogLevel.INFO, "dry point setup completed")
            else:
                report(LogLevel.INFO, "point setup completed")


def run_global_setup(data, execution):
    subspace = execution["subspace"]
    # a copy: the commands gathered below belong to this subspace only and
    # must not survive into the next preset's setup phase
    setup_commands = list(data["setup"]["global"])
    # gather setup commands from space
    for key, values in subspace.items():
        for value in values:
            if len(value["setup"]) > 0:
                for cmd in value["setup"]:
                    setup_commands.append(cmd)

    if execution["script"] is not None:
        if len(setup_commands) > 0:
            execution["script"].section("global setup")
            execution["script"].progress("global setup")
        report(LogLevel.INFO, "compiling global setup")
    elif execution["dry_run"]:
        report(LogLevel.INFO, "starting dry global setup")
    else:
        report(LogLevel.INFO, "starting global setup")

    errors = False
    for command in setup_commands:
        if execution["script"] is not None:
            execution["script"].command(
                substitute_global_yvars(command, subspace)
            )
        elif execution["dry_run"]:
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
    if execution["script"] is not None:
        pass
    elif execution["dry_run"]:
        report(LogLevel.INFO, "dry setup completed")
    else:
        report(LogLevel.INFO, "setup completed")


def run_setup(settings, data, execution):
    run_global_setup(data, execution)
    run_point_setup(data, execution)


def terminate(process, grace=2.0):
    """End a shell command and everything it started, politely then not."""
    try:
        group = os.getpgid(process.pid)
        os.killpg(group, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(group, signal.SIGKILL)
    except OSError:
        pass


class Trials:
    """The shell commands in flight, so that a kill can reach them.

    Each command runs in a session of its own, so terminating it reaches a whole
    pipeline rather than only the shell that started it. That also detaches it
    from yuclid's own terminal signals, which is why Ctrl-C is handled here
    instead of being left to the kernel.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.active = dict()
        self.abandoned = set()
        self.skipped = set()
        self.terminated = set()

    def spawn(self, key, command, env, cwd):
        process = subprocess.Popen(
            command,
            shell=True,
            env=env,
            cwd=cwd,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        with self.lock:
            self.active.setdefault(key, set()).add(process)
        try:
            stdout, stderr = process.communicate()
        finally:
            with self.lock:
                self.active.get(key, set()).discard(process)
                killed = process.pid in self.terminated
                self.terminated.discard(process.pid)
        return stdout, stderr, process.returncode, killed

    def kill(self, scope, keys=None):
        """Abandon what is running: this repetition of it, or the point.

        Returns the points it reached, so that whoever asked is told what was
        actually in flight rather than what they hoped was.
        """
        with self.lock:
            targets = [k for k in self.active if self.active[k]]
            if keys is not None:
                targets = [k for k in targets if k in keys]
            if scope == "point":
                self.abandoned.update(targets)
            else:
                self.skipped.update(targets)
            processes = [p for k in targets for p in self.active.get(k, ())]
            self.terminated.update(p.pid for p in processes)
        for process in processes:
            terminate(process)
        return targets, len(processes)

    def kill_everything(self):
        self.kill("point")

    def is_abandoned(self, key):
        with self.lock:
            return key in self.abandoned

    def take_skip(self, key):
        with self.lock:
            if key in self.skipped:
                self.skipped.discard(key)
                return True
            return False


def point_to_string(point):
    return ".".join([str(x["name"]) for x in point])


def metrics_to_string(metric_values):
    return " ".join([f"{m}={v}" for m, v in metric_values.items()])


def get_progress(unit, total):
    """Where a run has got to, counted in repetitions rather than points.

    With -r N every point is N units of work, so the counter and its total
    describe the whole experiment: a point already recorded by a previous run
    consumes its share of the count without being executed again.

    `unit` is always work finished, never work started, and the percentage is
    floored so that it reads 100% only once the run has genuinely got there.
    """
    if total <= 0:
        return "[{}/{}]".format(unit, total)
    return "[{}/{}] {}%".format(unit, total, 100 * unit // total)


def progress_units(execution, entry):
    """The total number of units, and the ones before this point.

    Both are read from the plan rather than from the size of the subspace, so
    they stay truthful once points have been dropped or added.
    """
    plan = execution["plan"]
    total, _ = plan.units()
    return total, plan.units_before(entry)


def run_point(settings, data, execution, writer, entry, file_lock=None):
    """Run a point, and leave the plan saying what became of it either way.

    A worker that raises would otherwise leave its entry marked as running for
    the rest of the run, which is the one status that cannot be true of it.
    """
    try:
        run_point_trials(settings, data, execution, writer, entry, file_lock)
    except Exception:
        execution["plan"].finish(entry, "failed")
        raise


def abandon_repetition(execution, entry, rep):
    """Give up the repetition in flight and let the point carry on."""
    execution["trials"].take_skip(entry["key"])
    execution["plan"].abandon_repetition(entry)
    report(
        LogLevel.WARNING,
        point_to_string(entry["point"]),
        "repetition abandoned, nothing recorded",
    )
    execution["progress"].emit(
        "point.killed",
        index=entry["seq"],
        key=list(entry["key"]),
        rep=rep,
        scope="rep",
    )


def run_point_trials(settings, data, execution, writer, entry, file_lock=None):
    os.makedirs(settings["trials_dir"], exist_ok=True)

    plan = execution["plan"]
    trials = execution["trials"]
    progress = execution["progress"]
    point = entry["point"]
    i = entry["seq"]

    point_map = {key: x for key, x in zip(execution["order"], point)}
    total, base = progress_units(execution, entry)
    report(
        LogLevel.INFO,
        get_progress(base + entry["done"], total),
        point_to_string(point),
        "started",
    )
    progress.emit(
        "point.started",
        index=i,
        key=list(entry["key"]),
        rep=entry["done"],
        completed=base + entry["done"],
        total=total,
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

    i_padded = str(i).zfill(plan.width())
    # with --resume, only the repetitions still missing are run, and their
    # numbering continues where the recorded ones left off
    requested = settings["repeat"]
    killed = False
    # a command that returned non-zero without our having killed it: only
    # --ignore-errors lets the run get past one, and the point should say so
    failed = False

    while entry["done"] < entry["target"] and not killed:
        if trials.is_abandoned(entry["key"]):
            break
        rep = entry["done"]
        rep_suffix = f"_rep{rep}" if requested > 1 else ""

        # every trial gets its own captures, and a metric must be evaluated
        # against the captures of the trial that enabled it
        metric_point_ids = dict()

        for j, trial in enumerate(compatible_trials):
            point_id = os.path.join(
                settings["trials_dir"],
                f"{i_padded}." + point_to_string(point) + f"{rep_suffix}_trial{j}",
            )

            for metric in compatible_metrics:
                if trial["metrics"] is None or metric["name"] in trial["metrics"]:
                    metric_point_ids[metric["name"]] = point_id

            command = substitute_global_yvars(trial["command"], execution["subspace"])
            command = substitute_point_yvars(command, point_map, point_id)
            stdout, stderr, returncode, was_killed = trials.spawn(
                entry["key"], command, execution["env"], settings["cwd"]
            )

            with open(f"{point_id}.out", "w") as output_file:
                if stdout:
                    output_file.write(stdout)

            with open(f"{point_id}.err", "w") as error_file:
                if stderr:
                    error_file.write(stderr)

            if was_killed:
                # a command we terminated ourselves did not fail: it was
                # abandoned, and saying otherwise would end the run
                killed = True
                break

            if returncode != 0:
                failed = True
                hint = "check the following files for more details:\n"
                hint += f"{point_id}.out\n{point_id}.err"
                report(
                    LogLevel.ERROR,
                    point_to_string(point),
                    f"failed trial command '{command}' (code {returncode})",
                    hint=hint,
                    fatal=settings["abort_on_error"],
                )

        if killed:
            if trials.is_abandoned(entry["key"]):
                break
            abandon_repetition(execution, entry, rep)
            killed = False
            continue

        collected_metrics = dict()
        for metric in compatible_metrics:
            metric_point_id = metric_point_ids[metric["name"]]
            command = substitute_global_yvars(metric["command"], execution["subspace"])
            command = substitute_point_yvars(command, point_map, metric_point_id)
            stdout, _, returncode, was_killed = trials.spawn(
                entry["key"], command, execution["env"], settings["cwd"]
            )
            text = stdout.strip()

            if was_killed:
                killed = True
                break

            if returncode != 0:
                failed = True
                hint = "check the following files for more details:\n"
                hint += f"{metric_point_id}.out\n{metric_point_id}.err\n"
                report(
                    LogLevel.ERROR,
                    point_to_string(point),
                    "metric {} failed with return code {}".format(
                        metric["name"], returncode
                    ),
                    hint=hint,
                    fatal=settings["abort_on_error"],
                )
            elif text == "":
                failed = True
                report(
                    LogLevel.ERROR,
                    point_to_string(point),
                    "metric {} generated an empty string".format(metric["name"]),
                    fatal=settings["abort_on_error"],
                )
            else:
                output_elements = re.split("[\n|\r| ]+", text)

                def int_or_float(x):
                    try:
                        return int(x)
                    except ValueError:
                        try:
                            return float(x)
                        except ValueError:
                            report(
                                LogLevel.ERROR,
                                "cannot parse result of metric {}".format(metric["name"]),
                                text,
                                hint="the command generated '{}'".format(text),
                                fatal=settings["abort_on_error"],
                            )
                            # the run carries on, so the sample has to be
                            # something: a gap, like any other missing one
                            return float("nan")

                collected_metrics[metric["name"]] = [
                    int_or_float(line) for line in output_elements
                ]

        # metrics may disagree on how many samples they produced: the shorter
        # ones leave gaps. Kept as plain lists so that an integer metric is not
        # widened to float by a float metric of the same point.
        samples = max((len(v) for v in collected_metrics.values()), default=0)
        padded = {
            name: values + [float("nan")] * (samples - len(values))
            for name, values in collected_metrics.items()
        }

        if not settings["fold"]:
            NaNs = [
                name
                for name, values in collected_metrics.items()
                if len(values) < samples
            ]
            if len(NaNs) > 0:
                report(
                    LogLevel.WARNING,
                    "the following metrics generated some NaNs",
                    " ".join(NaNs),
                )

        if killed or trials.take_skip(entry["key"]):
            # a kill that landed once the commands had already run: the
            # measurement is still one nobody asked to keep
            if trials.is_abandoned(entry["key"]):
                break
            abandon_repetition(execution, entry, rep)
            killed = False
            continue

        result = {k: x["name"] for k, x in point_map.items()}
        if file_lock is not None:
            file_lock.acquire()
        try:
            if settings["fold"]:
                result.update(padded)
                writer.write(result)
            else:
                for k in range(samples):
                    result.update({name: v[k] for name, v in padded.items()})
                    writer.write(result)
            writer.flush()
        finally:
            if file_lock is not None:
                file_lock.release()

        plan.repetition_done(entry)
        total, base = progress_units(execution, entry)
        completion = [
            get_progress(base + entry["done"], total),
            point_to_string(point),
            "failed" if failed else "completed",
        ]
        if len(collected_metrics) > 0:
            completion.append(metrics_to_string(collected_metrics))
        report(LogLevel.INFO, *completion)
        progress.emit(
            "point.finished",
            index=i,
            key=list(entry["key"]),
            rep=rep,
            repetitions=1,
            failed=failed,
            metrics={k: v for k, v in collected_metrics.items()},
            completed=base + entry["done"],
            total=total,
        )

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
                    "stddev",
                    metric_name,
                    f"{pd.Series(values).std():.3f}",
                )

    if killed or trials.is_abandoned(entry["key"]):
        report(LogLevel.WARNING, point_to_string(point), "abandoned")
        progress.emit("point.killed", index=i, key=list(entry["key"]), scope="point")
        plan.finish(entry, "killed")
    else:
        plan.finish(entry, "failed" if failed else "done")


def valid_conditions(point, order):
    point_context = {}
    yuclid = {name: x["value"] for name, x in zip(order, point)}
    point_context["yuclid"] = type("Yuclid", (), yuclid)()
    conditions = [x["condition"] for x in point if "condition" in x]
    return all(eval(c, point_context) for c in conditions)


_undecidable_conditions = set()


def valid_subpoint_conditions(point, suborder):
    # A point setup runs on a sub-point: a point restricted to the 'on'
    # dimensions, standing for every full point that shares those coordinates.
    # A condition mentioning a dimension outside 'on' therefore cannot be
    # decided here — it holds at some of those points and not at others — so
    # the sub-point is kept and the condition is left to the points themselves.
    point_context = {}
    yuclid = {name: x["value"] for name, x in zip(suborder, point)}
    point_context["yuclid"] = type("Yuclid", (), yuclid)()
    for condition in [x["condition"] for x in point if "condition" in x]:
        try:
            if not eval(condition, point_context):
                return False
        except (AttributeError, NameError):
            # not obvious, so say it out loud — but only once per condition
            if condition not in _undecidable_conditions:
                _undecidable_conditions.add(condition)
                report(
                    LogLevel.WARNING,
                    "cannot decide condition during point setup",
                    condition,
                    hint="it refers to dimensions outside 'on' ({}), where it holds "
                    "for some points and not for others. The setup command runs "
                    "anyway".format(", ".join(suborder)),
                )
            continue
    return True


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


def read_records(path, fmt):
    """Yield the records of a result dataset, or None for an unreadable one."""
    if fmt == "csv":
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                yield row
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None


def load_recorded_points(path, order, metric_names, fmt):
    """Count the records a previous run already wrote for each point.

    A record counts only if its non-metric fields are exactly this run's
    dimensions: anything else was produced by a different space and cannot be
    matched against the points about to run.
    """
    recorded = dict()
    if os.path.isdir(path):
        report(LogLevel.FATAL, "--resume needs a file, not a directory", path)
    if not os.path.isfile(path):
        report(
            LogLevel.WARNING,
            "nothing to resume: the file does not exist yet",
            path,
        )
        return recorded

    dimensions = set(order)
    foreign, unreadable = 0, 0
    try:
        for record in read_records(path, fmt):
            if record is None:
                unreadable += 1
                continue
            if set(record.keys()) - metric_names != dimensions:
                foreign += 1
                continue
            key = tuple(str(record[dim]) for dim in order)
            recorded[key] = recorded.get(key, 0) + 1
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        report(
            LogLevel.FATAL,
            "cannot read the file given to --resume",
            path,
            hint=str(e),
        )

    if unreadable > 0:
        report(
            LogLevel.FATAL,
            "the file given to --resume holds {} unreadable record(s)".format(
                unreadable
            ),
            path,
            hint="--resume expects a result dataset written by yuclid. "
            "Repair the file, or drop --resume to start a new one",
        )
    if len(recorded) == 0 and foreign > 0:
        report(
            LogLevel.FATAL,
            "no record in the file given to --resume belongs to this space",
            path,
            hint="its records must carry exactly these dimensions: {}".format(
                ", ".join(order)
            ),
        )
    if foreign > 0:
        report(
            LogLevel.WARNING,
            "ignoring records of a different space",
            foreign,
            hint="a record is matched only when its dimensions are exactly: {}".format(
                ", ".join(order)
            ),
        )
    report(
        LogLevel.INFO,
        "resuming from",
        path,
        "{} points already recorded".format(len(recorded)),
    )
    return recorded


def report_skipped(execution, entry):
    total, base = progress_units(execution, entry)
    report(
        LogLevel.INFO,
        get_progress(base + entry["target"], total),
        point_to_string(entry["point"]),
        "already recorded. Skipping",
    )


def run_subspace_trials(settings, data, execution):
    plan = execution["plan"]
    if execution["script"] is not None:
        from yuclid.compile import compile_subspace_trials

        compile_subspace_trials(settings, data, execution, execution["script"])
        return

    for entry in plan.skipped():
        report_skipped(execution, entry)

    if settings["dry_run"]:
        for entry in plan.pending():
            get_compatible_trials_and_metrics(data, entry["point"], execution)
            total, base = progress_units(execution, entry)
            report(
                LogLevel.INFO,
                get_progress(base + entry["target"], total),
                "dry run",
                point_to_string(entry["point"]),
            )
            execution["progress"].emit(
                "point.finished",
                index=entry["seq"],
                key=list(entry["key"]),
                repetitions=entry["target"],
                completed=base + entry["target"],
                total=total,
            )
            for _ in range(entry["target"] - entry["done"]):
                plan.repetition_done(entry)
            plan.finish(entry)
    else:
        output_dir = os.path.dirname(settings["output"])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        fresh = (
            not os.path.exists(settings["output"])
            or os.path.getsize(settings["output"]) == 0
        )
        with open(settings["output"], "a", newline="") as f:
            if settings["run_dir"] is not None:
                # only now does the file exist to be linked, so a run that never
                # gets this far leaves no empty dataset behind
                workspace.link_results(settings["run_dir"], settings["output"])
            writer = RecordWriter(
                f,
                settings["format"],
                record_columns(data, settings, execution["order"]),
                write_header=fresh,
            )
            if settings["parallel_trials"] != 0:
                if settings["parallel_trials"] < 0:
                    max_workers = execution["subspace_size"]
                else:
                    max_workers = settings["parallel_trials"]
                report(
                    LogLevel.INFO,
                    f"running trials in parallel with {max_workers} workers",
                )
                file_lock = threading.Lock()
                # a point is taken from the plan only when a worker is free for
                # it. `submit` queues without blocking, so without this the loop
                # would claim the whole plan at once and nothing could be paused
                # or dropped before it had already started
                slots = threading.Semaphore(max(1, max_workers))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, max_workers)
                ) as executor:
                    futures = []
                    for entry in plan.pending():
                        slots.acquire()
                        future = executor.submit(
                            run_point,
                            settings,
                            data,
                            execution,
                            writer,
                            entry,
                            file_lock,
                        )
                        future.add_done_callback(lambda _: slots.release())
                        futures.append(future)
                    for future in concurrent.futures.as_completed(futures):
                        exc = future.exception()
                        if exc is not None:
                            report(
                                LogLevel.ERROR,
                                "trial failed",
                                str(exc),
                                fatal=settings["abort_on_error"],
                            )
            else:
                for entry in plan.pending():
                    run_point(settings, data, execution, writer, entry)
                    writer.flush()


def validate_dimensions(subspace):
    undefined = [k for k, v in subspace.items() if v is None]
    if len(undefined) > 0:
        hint = "define dimensions with presets or select them with --select. "
        hint += "E.g. --select {}=value1,value2".format(undefined[0])
        report(LogLevel.FATAL, "dimensions undefined", ", ".join(undefined), hint=hint)


def prepare_subspace_execution(
    subspace, order, env, metrics, dry_run, repeat=1, recorded=None, script=None
):
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
    execution["recorded"] = recorded
    execution["script"] = script

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
    requested = args.inputs
    if requested is None:
        found = [name for name in DEFAULT_INPUTS if os.path.isfile(name)]
        if len(found) > 1:
            report(
                LogLevel.WARNING,
                "several default configurations found",
                ", ".join(found),
                hint="reading {}. Name the others with --inputs".format(found[0]),
            )
        requested = found[:1] or DEFAULT_INPUTS[:1]

    settings["inputs"] = []
    for file in requested:
        if not os.path.isfile(file):
            report(LogLevel.ERROR, f"'{file}' does not exist")
        else:
            settings["inputs"].append(file)
    # output
    # `--resume` continues the file named by --output
    settings["resume"] = None
    if args.resume:
        if args.output is None:
            report(
                LogLevel.FATAL,
                "--resume needs --output",
                hint="--resume continues the file given to --output, so there "
                "has to be one",
            )
        settings["resume"] = args.output

    settings["format"] = args.format
    if settings["format"] is None:
        # the extension of a name the user chose decides, jsonl otherwise
        named = args.output
        is_csv = named is not None and named.lower().endswith(".csv")
        settings["format"] = "csv" if is_csv else "jsonl"

    if settings["fold"] and settings["format"] == "csv":
        report(
            LogLevel.FATAL,
            "--fold cannot be written as CSV",
            hint="a CSV cell holds one value, not an array of samples. "
            "Drop --fold, or keep the default jsonl format",
        )

    settings["now"] = "{:%Y%m%d-%H%M%S}".format(datetime.now())
    filename = "{}.yuclid.{}".format(settings["now"], settings["format"])
    if args.output is None and args.output_dir is None:
        settings["output"] = filename
    elif args.output is not None and args.output_dir is not None:
        report(LogLevel.FATAL, "either --output or --output-dir must be specified")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        settings["output"] = os.path.join(args.output_dir, filename)
    else:
        settings["output"] = args.output

    if args.compile is not None:
        ignored = [
            name
            for name, active in [
                ("--dry-run", args.dry_run),
                ("--parallel-trials", args.parallel_trials != 0),
                ("--resume", args.resume),
            ]
            if active
        ]
        if len(ignored) > 0:
            report(
                LogLevel.WARNING,
                "ignoring {}".format(", ".join(ignored)),
                hint="--compile only writes a script: it runs nothing, and the "
                "script it writes is sequential",
            )

    settings["timing"] = {"setup": 0.0, "experiments": 0.0}
    settings["cwd"] = os.getcwd()
    build_run_directory(settings, args)
    settings["steering"] = Steering(settings, settings["progress"])

    report(LogLevel.INFO, "working directory", settings["cwd"])
    report(LogLevel.INFO, "input configurations", ", ".join(requested))
    report(LogLevel.INFO, "output data", settings["output"])
    report(LogLevel.INFO, "temp directory", settings["trials_dir"])
    return settings


def build_run_directory(settings, args):
    """Give the run a directory of its own to record itself in.

    Compiling writes a script and runs nothing, so it gets no directory: there
    would be no run to describe.
    """
    settings["run_dir"] = None
    settings["run_id"] = None
    settings["root"] = None
    settings["progress"] = workspace.Progress(None)
    settings["trials"] = Trials()

    if args.compile is not None:
        settings["trials_dir"] = os.path.join(
            args.temp_dir or os.path.join(workspace.DIRNAME, workspace.TRIALS),
            settings["now"],
        )
        return

    if args.name is not None:
        # checked before a directory is claimed, so a name that cannot be used
        # leaves no half-started run behind
        try:
            workspace.check_name(args.name)
        except ValueError as e:
            report(LogLevel.FATAL, "invalid --name", str(e))

    settings["root"] = workspace.open_root()
    settings["run_id"], settings["run_dir"] = workspace.create_run(
        settings["root"],
        settings["now"],
        argv=sys.argv[1:],
        inputs=settings["inputs"],
        output=os.path.abspath(settings["output"]),
        replay_of=args.replay,
    )
    if args.name is not None:
        workspace.write_name(settings["run_dir"], args.name)
    build_replay_schedule(settings, args)

    if args.temp_dir is None:
        settings["trials_dir"] = os.path.join(settings["run_dir"], workspace.TRIALS)
    else:
        settings["trials_dir"] = os.path.join(args.temp_dir, settings["now"])
    os.makedirs(settings["trials_dir"], exist_ok=True)
    settings["progress"] = workspace.Progress(
        os.path.join(settings["run_dir"], workspace.PROGRESS)
    )


def build_replay_schedule(settings, args):
    """Work out what the run being replayed was told, and write it down here.

    The schedule lives in the new run's directory, so a replay says what it
    replayed and can itself be replayed in turn.
    """
    settings["schedule"] = []
    if args.replay is None:
        return

    source = workspace.run_directory(settings["root"], args.replay)
    if workspace.read_manifest(source) is None:
        report(
            LogLevel.FATAL,
            "no such run",
            args.replay,
            hint="`yuclid runs` lists what there is to replay",
        )

    schedule = derive_schedule(source)
    with open(os.path.join(settings["run_dir"], workspace.REPLAY), "w") as f:
        json.dump({"replay_of": args.replay, "schedule": schedule}, f, indent=2)
        f.write("\n")
    settings["schedule"] = schedule

    report(
        LogLevel.INFO,
        "replaying",
        args.replay,
        "{} steering operation(s)".format(len(schedule)),
    )


def normalize_point_setup(point_setup, space):
    if isinstance(point_setup, str):
        # a bare command is a point setup of one item
        point_setup = [point_setup]

    if not isinstance(point_setup, list):
        report(LogLevel.FATAL, "point setup must be a string or a list")

    normalized_items = []
    for item in point_setup:
        if isinstance(item, (str, list)):
            # an item that is just a command runs on the whole space
            item = {"command": item}
        if not isinstance(item, dict):
            report(LogLevel.FATAL, "point setup must be a string or a list")

        unexpected = [x for x in item if x not in ["command", "on", "parallel"]]
        if len(unexpected) > 0:
            report(
                LogLevel.WARNING,
                "point setup item has unexpected fields",
                ", ".join(unexpected),
                hint="fields: 'command', 'on', 'parallel'",
            )

        if "command" not in item:
            report(
                LogLevel.FATAL,
                "point setup item must have 'command' field",
            )

        normalized_item = {
            "command": normalize_command(item["command"]),
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


def make_expander(settings, data, execution):
    """Turn coordinates named by a steering command into points to run.

    A value the space never had is admitted here, which is what lets a run be
    extended rather than only pruned. Its own setup commands run at that moment,
    since the setup phase is long past; the global setup is not run again.
    """

    def expand(coords):
        subspace = execution["subspace"]
        order = execution["order"]

        for dim, names in coords.items():
            known = {str(p["name"]) for p in subspace[dim]}
            fresh = [normalize_point(name) for name in names if name not in known]
            if len(fresh) > 0:
                subspace[dim] = list(subspace[dim]) + fresh
                for value in fresh:
                    run_value_setup(execution, value)

        axes = []
        for dim in order:
            values = subspace[dim]
            if dim in coords:
                values = [p for p in values if str(p["name"]) in coords[dim]]
            axes.append(values)

        execution["subspace_values"] = {
            key: [x["value"] for x in subspace[key]] for key in subspace
        }
        execution["subspace_names"] = {
            key: [x["name"] for x in subspace[key]] for key in subspace
        }
        return [p for p in itertools.product(*axes) if valid_conditions(p, order)]

    return expand


def run_value_setup(execution, value):
    for command in value["setup"]:
        command = substitute_global_yvars(command, execution["subspace"])
        report(LogLevel.INFO, "setup for a value just added", command)
        result = subprocess.run(
            command, shell=True, universal_newlines=True, env=execution["env"]
        )
        if result.returncode != 0:
            report(LogLevel.WARNING, "setup", f"'{command}'", f"returned {result.returncode}")


def describe_op(op, effect):
    """What an operation did, in the words the person who asked would use."""
    kind = op.get("op")
    if kind == "drop":
        return "dropped {} point(s), {} unit(s) remain".format(
            effect.get("dropped", 0), effect["total"] - effect["completed"]
        )
    if kind == "add":
        return "added {} point(s), restored {}".format(
            effect.get("added", 0), effect.get("restored", 0)
        )
    if kind == "kill":
        return "killed {} point(s), {} command(s)".format(
            effect.get("killed", 0), effect.get("commands", 0)
        )
    if kind == "stop":
        return "stopping, {} point(s) abandoned".format(effect.get("abandoned", 0))
    if kind == "repeat":
        return "{} point(s) now ask for {} repetition(s)".format(
            effect.get("retargeted", 0), effect.get("repeat")
        )
    if kind == "order":
        return "reordered {} pending point(s)".format(effect.get("reordered", 0))
    return kind


class Steering:
    """What the run answers when somebody asks it to change course.

    A run with several presets executes one plan after another, so the current
    one is held here rather than passed around: a command always addresses what
    is running now.
    """

    def __init__(self, settings, progress):
        self.settings = settings
        self.progress = progress
        self.execution = None
        self.server = None
        self.lock = threading.Lock()

    def attach(self, execution):
        self.execution = execution

    def start(self, path):
        self.server = control.Server(path, self.handle)
        failure = self.server.start()
        if failure is not None:
            # a run must never be lost to a socket that could not be bound
            report(LogLevel.WARNING, "this run cannot be steered", failure)

    def close(self):
        # a request being answered outlives the run by a moment: killing the
        # last point ends the run, and whoever asked for it deserves the answer
        with self.lock:
            if self.server is not None:
                self.server.close()
                self.server = None

    def handle(self, message):
        with self.lock:
            if self.server is None:
                raise control.ControlError("the run has ended")
            return self.dispatch(message)

    def dispatch(self, message):
        kind = message.get("op")
        if kind == "status":
            return self.status()
        if self.execution is None:
            raise control.ControlError("this run has not started its experiments yet")
        if kind == "kill":
            effect = self.kill(message)
        else:
            effect = self.execution["plan"].apply(message)
        report(LogLevel.INFO, "control", describe_op(message, effect))
        self.progress.emit(
            "op.applied", op=message, effect=effect, total=effect.get("total")
        )
        # the plan has just changed shape, so record the new one: readers take
        # the last snapshot rather than replaying what every operation meant
        plan = self.execution["plan"]
        self.progress.emit(
            "plan", preset=self.execution.get("preset"), **plan.snapshot()
        )
        return effect

    def status(self):
        if self.execution is None:
            return {"state": "preparing"}
        plan = self.execution["plan"]
        total, completed = plan.units()
        return {
            "state": "paused" if plan.paused else "running",
            "completed": completed,
            "total": total,
            "in_flight": [list(e["key"]) for e in plan.running()],
        }

    def kill(self, message):
        scope = message.get("scope")
        if scope not in ("point", "rep"):
            raise control.ControlError(
                "kill needs a scope: 'point' abandons it entirely, "
                "'rep' only the repetition in flight"
            )
        plan = self.execution["plan"]
        keys = None
        if message.get("coords"):
            coords = message["coords"]
            unknown = [d for d in coords if d not in plan.order]
            if unknown:
                raise control.ControlError(
                    "unknown dimension(s) {}".format(", ".join(unknown))
                )
            keys = {
                e["key"]
                for e in plan.running()
                if all(
                    e["key"][plan.order.index(dim)] in values
                    for dim, values in coords.items()
                )
            }
        targets, commands = self.settings["trials"].kill(scope, keys)
        if len(targets) == 0:
            raise control.ControlError("nothing is running to kill")
        total, completed = plan.units()
        return {
            "killed": len(targets),
            "commands": commands,
            "scope": scope,
            "points": [list(k) for k in targets],
            "completed": completed,
            "total": total,
        }


def derive_schedule(directory):
    """Turn what a run was told to do into a schedule another run can follow.

    Boundary operations are anchored to the number of repetitions completed
    when they were applied, not to the clock, so a replay lands them in the same
    place however fast the machine is. A kill is anchored to the point and
    repetition it interrupted, and to how far into it the interruption came.
    """
    schedule = []
    completed = 0
    started = dict()

    for record in workspace.read_progress(directory):
        kind = record["type"]
        if kind == "point.started":
            started[(tuple(record["key"]), record.get("rep", 0))] = record["time"]
        elif kind == "point.finished":
            completed = record.get("completed", completed)
        elif kind == "point.killed":
            key = tuple(record["key"])
            rep = record.get("rep", 0)
            began = started.get((key, rep))
            schedule.append(
                {
                    "at_point": list(key),
                    "at_rep": rep,
                    "after_seconds": round(record["time"] - began, 3) if began else 0.0,
                    "op": {"op": "kill", "scope": record.get("scope", "point")},
                }
            )
        elif kind == "op.applied" and record["op"].get("op") != "kill":
            # a kill is replayed from what it actually interrupted, above
            schedule.append({"after_units": completed, "op": record["op"]})

    return schedule


class Replay:
    """Re-apply a recorded schedule to the run now under way.

    It watches rather than being called: the point loop knows nothing about
    replaying, and a kill has to fire in the middle of a point anyway.
    """

    def __init__(self, schedule, steering):
        self.pending = list(schedule)
        self.steering = steering
        self.thread = None
        self.stopped = False

    def start(self):
        if len(self.pending) == 0:
            return
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def close(self):
        self.stopped = True

    def _watch(self):
        while not self.stopped and len(self.pending) > 0:
            item = self.pending[0]
            if "after_units" in item:
                if self._units() >= item["after_units"]:
                    self.pending.pop(0)
                    self._apply(item["op"])
            elif self._kill_due(item):
                self.pending.pop(0)
            time.sleep(0.05)

    def _units(self):
        execution = self.steering.execution
        if execution is None:
            return -1
        return execution["plan"].units()[1]

    def _kill_due(self, item):
        execution = self.steering.execution
        if execution is None:
            return False
        plan = execution["plan"]
        entry = plan.in_flight(tuple(item["at_point"]))
        if entry is None or entry["done"] != item["at_rep"]:
            # the point may already be past: a kill that has nothing to
            # interrupt is a kill that does nothing, and says so
            if self._units() > 0 and plan.units()[1] >= plan.units()[0]:
                report(
                    LogLevel.WARNING,
                    "replayed kill found nothing to interrupt",
                    ".".join(item["at_point"]),
                )
                return True
            return False
        time.sleep(item.get("after_seconds", 0.0))
        if plan.in_flight(tuple(item["at_point"])) is None:
            report(
                LogLevel.WARNING,
                "replayed kill arrived after the command had finished",
                ".".join(item["at_point"]),
            )
            return True
        self._apply(dict(item["op"], coords=coords_of(plan.order, item["at_point"])))
        return True

    def _apply(self, op):
        try:
            self.steering.handle(op)
        except control.ControlError as e:
            report(LogLevel.WARNING, "replayed operation declined", str(e))


def coords_of(order, key):
    return {dim: [value] for dim, value in zip(order, key)}


def run_experiments(
    settings, data, order, env, preset_name=None, recorded=None, script=None
):
    if preset_name is None:
        subspace = data["space"].copy()
    else:
        subspace = apply_preset(data, preset_name)

    subspace = apply_user_selectors(settings, subspace)
    validate_dimensions(subspace)
    print_subspace(subspace)
    execution = prepare_subspace_execution(
        subspace, order, env, metrics=settings["metrics"], dry_run=settings["dry_run"],
        repeat=settings["repeat"], recorded=recorded, script=script,
    )
    validate_execution(execution, data)

    execution["trials"] = settings["trials"]
    execution["progress"] = settings["progress"]
    execution["preset"] = preset_name
    execution["plan"] = Plan(
        order,
        execution["subspace_points"],
        settings["repeat"],
        recorded=recorded,
        expand=make_expander(settings, data, execution),
    )
    settings["steering"].attach(execution)
    execution["progress"].emit(
        "plan", preset=preset_name, **execution["plan"].snapshot()
    )

    if not settings["no_setup"]:
        started = time.monotonic()
        run_setup(settings, data, execution)
        settings["timing"]["setup"] += time.monotonic() - started
    else:
        report(LogLevel.INFO, "skipping setup phase")
    started = time.monotonic()
    try:
        run_subspace_trials(settings, data, execution)
    finally:
        # what every point came to, recorded even if the run is cut short: the
        # last snapshot is what a reader takes as the truth
        execution["progress"].emit(
            "plan", preset=preset_name, **execution["plan"].snapshot()
        )
    settings["timing"]["experiments"] += time.monotonic() - started


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


def format_duration(seconds):
    if seconds < 10:
        return "{:.1f}s".format(seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def launch(args):
    settings = build_settings(args)
    if settings["run_dir"] is None:
        execute(settings)
        return

    steering = settings["steering"]
    steering.start(os.path.join(settings["run_dir"], workspace.CONTROL))
    previous = install_interrupt_handler(settings["trials"])
    replay = Replay(settings["schedule"], steering)
    replay.start()

    state = workspace.FAILED
    try:
        execute(settings)
        state = settings.get("final_state", workspace.FINISHED)
    except KeyboardInterrupt:
        state = workspace.INTERRUPTED
        raise
    except SystemExit as e:
        state = workspace.FINISHED if not e.code else workspace.FAILED
        raise
    finally:
        replay.close()
        settings["progress"].emit("run.finished", state=state)
        steering.close()
        settings["progress"].close()
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
        # nothing is watching a run, so it records its own ending
        workspace.set_state(settings["run_dir"], state)


def install_interrupt_handler(trials):
    """Take the trials down with the run when Ctrl-C arrives.

    Each trial runs in a session of its own so that a kill can reach a whole
    pipeline, which also means the terminal's own interrupt no longer reaches
    it. Forwarding it here keeps Ctrl-C meaning what it always meant.
    """

    def handler(signum, frame):
        trials.kill_everything()
        raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGINT, handler)
    except ValueError:
        return None


def execute(settings):
    started = time.monotonic()
    data = aggregate_input_data(settings)
    validate_settings(data, settings)
    env = build_environment(settings, data)
    order = define_order(settings, data)
    validate_yvars_in_env(data["space"], data["env"])
    validate_yvars_in_setup(data["space"], data["setup"])
    validate_yvars_in_trials(data["space"], data["trials"])

    validate_presets(settings, data)

    if settings["compile"] is not None:
        from yuclid.compile import compile_experiments

        compile_experiments(settings, data, order, env)
        report(
            LogLevel.INFO, "finished in", format_duration(time.monotonic() - started)
        )
        return

    recorded = None
    if settings["resume"] is not None:
        metric_names = {m["name"] for m in data["metrics"]}
        recorded = load_recorded_points(
            settings["resume"], order, metric_names, settings["format"]
        )
        if settings["repeat"] > 1 and not settings["fold"]:
            report(
                LogLevel.WARNING,
                "counting one recorded repetition per record",
                hint="a metric emitting several samples writes one record per "
                "sample, so a point may look more repeated than it is. Use "
                "--fold for an exact count",
            )

    if len(settings["presets"]) > 0:
        for preset_name in settings["presets"]:
            report(LogLevel.INFO, "running preset", preset_name)
            run_experiments(settings, data, order, env, preset_name, recorded)
            report(LogLevel.INFO, "completed preset", preset_name)
    else:
        run_experiments(settings, data, order, env, preset_name=None, recorded=recorded)

    report(
        LogLevel.INFO,
        "finished in",
        "{} (setup {}, experiments {})".format(
            format_duration(time.monotonic() - started),
            format_duration(settings["timing"]["setup"]),
            format_duration(settings["timing"]["experiments"]),
        ),
    )

    if not settings["dry_run"]:
        metric_names = {m["name"] for m in data["metrics"]}
        hint = "use `yuclid plot {}` to analyze the results".format(settings["output"])
        report(LogLevel.INFO, "output data written to", settings["output"], hint=hint)

from yuclid.log import report, LogLevel
from yuclid.plot import (
    validate_files,
    locate_files,
    generate_dataframe,
    combine_dimensions,
    explode_array_metrics,
)
import urllib.request
import urllib.error
import json
import re
import types
import sys
import termios
import select
import tty
import os
import pandas as pd


_SUGGEST_SYSTEM_PROMPT = """\
You are a data visualization assistant for the yuclid benchmarking tool.
You will be given a description of an experiment dataset and the plotting capabilities of yuclid.
Your task is to suggest 3 to 5 meaningful and distinct visualizations.

IMPORTANT RULES:
- Only use dimension names and metric names EXACTLY as they appear in the schema.
- Only use flags and values that are valid per the capability description.
- Each suggestion must be a valid yuclid plot/tplot command argument list.
- Do NOT invent new columns or flag names.
- Respond ONLY with a JSON array. No explanation text outside the JSON.

Each element of the array must be a JSON object with these fields:
  "description": <one-sentence human-readable summary>,
  "args": <array of CLI argument strings, as if passed after the filename>

Example of valid output:
[
  {
    "description": "Compare time.real across compression levels, grouped by size",
    "args": ["-x", "compression", "-y", "time.real", "-z", "size"]
  },
  {
    "description": "Show time.real as a line plot across nthreads, grouped by impl",
    "args": ["-x", "nthreads", "-y", "time.real", "-z", "impl", "--lines"]
  },
  {
    "description": "Relative speedup over baseline impl, normalized per x-group",
    "args": ["-x", "size", "-y", "time.real", "-z", "impl", "--z-norm", "impl=ref", "--geomean"]
  }
]
"""

_TRANSLATE_SYSTEM_PROMPT = """\
You are a data visualization assistant for the yuclid benchmarking tool.
The user will describe in natural language what they want to visualize.
Your task is to translate their request into exactly ONE valid yuclid plot command.

IMPORTANT RULES:
- Only use dimension names and metric names EXACTLY as they appear in the schema.
- Only use flags and values that are valid per the capability description.
- The result must be a valid yuclid plot/tplot command argument list.
- Do NOT invent new columns or flag names.
- Respond ONLY with a JSON array containing exactly one element. No explanation text outside the JSON.

The single element must be a JSON object with these fields:
  "description": <one-sentence human-readable summary of what the plot shows>,
  "args": <array of CLI argument strings, as if passed after the filename>

Example of valid output:
[
  {
    "description": "Relative speedup over baseline impl, normalized per x-group",
    "args": ["-x", "size", "-y", "time.real", "-z", "impl", "--z-norm", "impl=ref", "--geomean"]
  }
]
"""


def inspect_data(file_path):
    args = types.SimpleNamespace(
        files=[file_path],
        filter=None,
        no_merge_inputs=False,
        array_reduce=None,
        combine=[],
        y=[],
    )
    ctx = {"args": args}
    validate_files(ctx)
    locate_files(ctx)
    generate_dataframe(ctx)
    combine_dimensions(ctx)
    explode_array_metrics(ctx)

    df = ctx["df"]
    metrics = []
    dimensions = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col)
        else:
            dimensions.append(col)

    dim_info = []
    for col in dimensions:
        vals = df[col].unique().tolist()
        truncated = vals[:20]
        entry = {"name": col, "values": truncated}
        if len(vals) > 20:
            entry["values"].append("...")
        dim_info.append(entry)

    return {"dimensions": dim_info, "metrics": metrics}


def list_models(base_url="http://localhost:11434"):
    req = urllib.request.Request(f"{base_url}/api/tags")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
    return [m["name"] for m in data.get("models", [])]


def select_model(requested, base_url="http://localhost:11434"):
    models = list_models(base_url)
    if not models:
        report(LogLevel.FATAL, "Ollama has no models loaded",
               hint="run: ollama pull <model>")
    if requested is not None:
        if requested not in models:
            report(LogLevel.WARNING,
                   f"model '{requested}' not found, using '{models[0]}'")
            return models[0]
        return requested
    preferred = ["gemma", "llama", "mistral", "phi", "qwen"]
    for kw in preferred:
        for m in models:
            if kw in m.lower():
                return m
    return models[0]


def chat(model, messages, base_url="http://localhost:11434"):
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def build_capability_description(info):
    lines = ["## Data Schema", "", "Dimensions (categorical):"]
    for d in info["dimensions"]:
        lines.append(f"  - {d['name']}: {d['values']}")
    lines += ["", "Metrics (numeric):"]
    for m in info["metrics"]:
        lines.append(f"  - {m}")
    lines += [
        "",
        "## Yuclid Plot Capabilities",
        "",
        "Required flags:",
        "  -x <dim>      X-axis (any dimension name from above)",
        "  -y <metric>   Y-axis (one metric name from above)",
        "  -z <dim>      Grouping/legend dimension (must differ from -x)",
        "",
        "Optional flags:",
        "  --lines",
        "      Line plot instead of bar chart. Best when x-axis is numeric or ordered.",
        "      Example: -x nthreads -y time -z impl --lines",
        "",
        "  --x-norm <x-dim>=<val>",
        "      Normalize all y-values relative to a baseline x-group value.",
        "      The dim in <x-dim>=<val> MUST match the value given to -x.",
        "      Use to show relative speedup/slowdown across the x-axis.",
        "      Example: -x size -y time -z compression --x-norm size=small",
        "      (every bar is divided by the value at size=small within its z-group)",
        "",
        "  --z-norm <z-dim>=<val>",
        "      Normalize all y-values relative to a baseline z-group value.",
        "      The dim in <z-dim>=<val> MUST match the value given to -z.",
        "      Use to compare groups relative to a reference group.",
        "      Example: -x nthreads -y time -z impl --z-norm impl=ref",
        "      (every bar is divided by the value of impl=ref at the same x)",
        "",
        "  --ref-norm <x-dim>=<val> <z-dim>=<val>",
        "      Normalize all y-values relative to a single fixed reference point.",
        "      Requires exactly two key=val pairs: one for the x dimension and one for the z dimension.",
        "      The dims MUST match the values given to -x and -z respectively.",
        "      Example: -x size -y time -z impl --ref-norm size=small impl=ref",
        "      (all bars divided by the single value at size=small, impl=ref)",
        "",
        "  --geomean",
        "      Append a geometric mean summary bar at the end (bar charts only).",
        "      Most useful when combined with a normalization flag.",
        "      Example: -x size -y time -z impl --z-norm impl=ref --geomean",
        "",
        "  -f <dim>=<val>",
        "      Filter rows to a specific dimension value before plotting.",
        "      Use when the data has many dimensions and you want to focus on one slice.",
        "      Example: -x nthreads -y time -z malloc -f program=bc",
        "",
        "Constraints:",
        "  -x and -z must be different dimensions",
        "  --x-norm, --z-norm, --ref-norm are mutually exclusive",
        "  --geomean and --lines cannot be combined",
        "  The value used in --x-norm / --z-norm / --ref-norm must exist in the data",
    ]
    return "\n".join(lines)


def parse_suggestions(raw):
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw.strip(), flags=re.MULTILINE)
    idx = raw.find("[")
    if idx == -1:
        report(LogLevel.FATAL, "LLM did not return a JSON array",
               hint="try a more capable model with --model")
    try:
        suggestions = json.loads(raw[idx:])
    except json.JSONDecodeError as e:
        report(LogLevel.FATAL, f"failed to parse LLM response as JSON: {e}",
               hint="try a more capable model with --model")
    valid = [
        s for s in suggestions
        if isinstance(s, dict)
        and isinstance(s.get("description"), str)
        and isinstance(s.get("args"), list)
    ]
    if not valid:
        report(LogLevel.FATAL, "no valid suggestions found in LLM response",
               hint="try a more capable model with --model")
    return valid


def validate_suggestion(suggestion, info):
    known_dims = {d["name"] for d in info["dimensions"]}
    known_metrics = set(info["metrics"])
    known_all = known_dims | known_metrics
    warnings = []
    args = suggestion["args"]
    for i, token in enumerate(args):
        if token in ("-x", "-y", "-z") and i + 1 < len(args):
            name = args[i + 1]
            if name not in known_all:
                warnings.append(f"unknown column '{name}' for {token}")
    return warnings


_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_CYAN    = "\033[96m"
_YELLOW  = "\033[93m"
_UP      = "\033[1A"
_CLR     = "\033[2K"


def _getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                ch += os.read(fd, 2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _arrow_menu(title, items, detail_fn=None):
    """Display an arrow-key navigable menu. Returns selected index."""
    n = len(items)
    idx = 0
    # fixed line count per render: 1 blank + 1 title + 1 blank + n items + (n detail rows if detail_fn)
    lines_per_render = 3 + n + (n if detail_fn else 0)

    def _render():
        out = [f"\n{_BOLD}{title}{_RESET}\n"]
        for i, item in enumerate(items):
            if i == idx:
                out.append(f"  {_CYAN}▶ {item}{_RESET}")
            else:
                out.append(f"    {_DIM}{item}{_RESET}")
            if detail_fn:
                detail = detail_fn(i) if i == idx else ""
                out.append(f"    {_DIM}{detail}{_RESET}" if detail else "")
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()

    def _erase():
        sys.stdout.write(f"\033[{lines_per_render}A\033[0J")
        sys.stdout.flush()

    _render()
    while True:
        ch = _getch()
        if ch in (b"\x1b[A", b"\x1b[D"):   # up / left
            new_idx = (idx - 1) % n
        elif ch in (b"\x1b[B", b"\x1b[C"): # down / right
            new_idx = (idx + 1) % n
        elif ch in (b"\r", b"\n", b" "):    # enter / space
            sys.stdout.write("\n")
            return idx
        elif ch == b"q":
            sys.stdout.write("\n")
            print("Aborted.")
            sys.exit(0)
        else:
            continue
        _erase()
        idx = new_idx
        _render()


def prompt_user(file_path, suggestions):
    labels = [s["description"] for s in suggestions]

    idx = _arrow_menu("Suggested Visualizations  (↑↓ navigate, Enter select, q quit)",
                      labels)
    chosen = suggestions[idx]

    renderer_idx = _arrow_menu("", ["plot  (GUI)", "tplot  (terminal)"])
    renderer = "tplot" if renderer_idx == 1 else "plot"

    cmd = f"yuclid {renderer} {file_path} " + " ".join(chosen["args"])
    print(f"\n{_BOLD}Command:{_RESET} {_CYAN}{cmd}{_RESET}\n")

    return chosen, renderer


def execute_suggestion(file_path, suggestion, renderer):
    import yuclid.cli
    import yuclid.plot
    import yuclid.tplot

    full_args = [renderer, file_path] + suggestion["args"]
    try:
        args = yuclid.cli.get_parser().parse_args(full_args)
    except SystemExit:
        report(LogLevel.FATAL, "could not parse suggested arguments",
               hint="args: " + " ".join(suggestion["args"]))

    if renderer == "plot":
        yuclid.plot.launch(args)
    else:
        yuclid.tplot.launch(args)


def _ask_llm(model, base_url, system_prompt, description, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            "Here is the data and capability description:\n\n"
            + description + "\n\n" + user_prompt},
    ]
    report(LogLevel.INFO, "generating parameters...")
    try:
        return chat(model, messages, base_url)
    except urllib.error.URLError as e:
        report(LogLevel.FATAL, f"Ollama request failed: {e}")


def _get_valid_suggestions(raw, info):
    suggestions = parse_suggestions(raw)
    valid = []
    for s in suggestions:
        warns = validate_suggestion(s, info)
        if warns:
            for w in warns:
                report(LogLevel.WARNING, f"skipping '{s['description']}': {w}")
        else:
            valid.append(s)
    if not valid:
        report(LogLevel.FATAL, "all LLM suggestions referred to invalid columns",
               hint="try a more capable model with --model")
    return valid


def launch(args):
    base_url = args.ollama_url
    file_path = args.files[0]
    if len(args.files) > 1:
        report(LogLevel.WARNING, "panorama uses only the first file; ignoring the rest")

    try:
        model = select_model(args.model, base_url)
    except urllib.error.URLError:
        report(LogLevel.FATAL, f"cannot connect to Ollama at {base_url}",
               hint="make sure Ollama is running: ollama serve")

    report(LogLevel.INFO, f"using model: {model}")

    info = inspect_data(file_path)
    report(LogLevel.INFO,
           f"found {len(info['dimensions'])} dimensions and {len(info['metrics'])} metrics")

    description = build_capability_description(info)

    mode = _arrow_menu("How do you want to explore the data?",
                       ["Suggest automatically", "Describe what you want"])

    if mode == 0:
        raw = _ask_llm(model, base_url, _SUGGEST_SYSTEM_PROMPT,
                       description, "Propose 3–5 visualizations.")
        valid = _get_valid_suggestions(raw, info)
        chosen, renderer = prompt_user(file_path, valid)
    else:
        print(f"\n{_BOLD}Describe what you want to see:{_RESET} ", end="", flush=True)
        user_prompt = input().strip()
        if not user_prompt:
            report(LogLevel.FATAL, "no prompt provided")
        raw = _ask_llm(model, base_url, _TRANSLATE_SYSTEM_PROMPT,
                       description, f"User request: {user_prompt}")
        valid = _get_valid_suggestions(raw, info)
        chosen = valid[0]
        print(f"\n{_DIM}{' '.join(chosen['args'])}{_RESET}")
        renderer_idx = _arrow_menu("", ["plot  (GUI)", "tplot  (terminal)"])
        renderer = "tplot" if renderer_idx == 1 else "plot"
        cmd = f"yuclid {renderer} {file_path} " + " ".join(chosen["args"])
        print(f"\n{_BOLD}Command:{_RESET} {_CYAN}{cmd}{_RESET}\n")

    execute_suggestion(file_path, chosen, renderer)

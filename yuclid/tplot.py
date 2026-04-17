from yuclid.log import report, LogLevel
from yuclid.plot import (
    validate_files,
    locate_files,
    generate_dataframe,
    combine_dimensions,
    generate_derived_metrics,
    explode_array_metrics,
    validate_args,
    reorder_and_numericize,
    rescale,
    generate_space,
    compute_ylimits,
    get_current_config,
    get_projection,
    group_normalization,
    ref_normalization,
    get_config_name,
)
import yuclid.spread as spread
import plotext as ptx
import pandas as pd
import numpy as np
import scipy.stats
import colorsys
import pathlib
import random
import termios
import select
import json
import sys
import tty
import re
import os


_CACHE_PATH = pathlib.Path.home() / ".yuclid" / "tplot.json"


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_REVERSE = "\033[7m"


def _new_colors(n):
    """Generate n visually distinct RGB colors using a random hue offset."""
    offset = random.random()
    colors = []
    for i in range(n):
        h = (offset + i / n) % 1.0
        s = random.uniform(0.55, 0.95)
        v = random.uniform(0.75, 1.00)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _new_bg():
    """Generate random (canvas_rgb, axes_rgb, ticks_rgb) for the background."""
    h = random.random()
    # canvas: dark, slightly saturated
    r, g, b = colorsys.hsv_to_rgb(h, random.uniform(0.3, 0.6), random.uniform(0.08, 0.22))
    canvas = (int(r * 255), int(g * 255), int(b * 255))
    # axes: same hue family, slightly different shade
    r, g, b = colorsys.hsv_to_rgb((h + 0.04) % 1.0, random.uniform(0.2, 0.45), random.uniform(0.15, 0.30))
    axes = (int(r * 255), int(g * 255), int(b * 255))
    # ticks: complementary hue, high value — readable against dark bg
    r, g, b = colorsys.hsv_to_rgb((h + 0.5) % 1.0, random.uniform(0.1, 0.3), random.uniform(0.85, 1.0))
    ticks = (int(r * 255), int(g * 255), int(b * 255))
    return canvas, axes, ticks


def _load_cache():
    try:
        data = json.loads(_CACHE_PATH.read_text())
        return {
            "colors": [tuple(c) for c in data["colors"]] if "colors" in data else None,
            "bg": tuple(tuple(c) for c in data["bg"]) if "bg" in data else None,
        }
    except Exception:
        return {}


def _save_cache(ctx):
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if ctx.get("colors"):
            data["colors"] = ctx["colors"]
        if ctx.get("bg"):
            data["bg"] = ctx["bg"]
        _CACHE_PATH.write_text(json.dumps(data))
    except Exception:
        pass


def _getch():
    """Read one keypress from stdin, handling arrow-key escape sequences."""
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


def _status_height(ctx):
    """Return the number of terminal rows consumed by the status block."""
    # 1 blank line + 1 hint line always present
    # +1 extra to account for a two-row terminal prompt after exit
    rows = 3
    if len(ctx["y_dims"]) > 1:
        rows += 1
    if ctx["free_dims"]:
        rows += 1
    return rows


def _render(ctx):
    """Clear the terminal and re-draw the chart plus status line."""
    print("\033[H\033[J", end="", flush=True)
    term = os.get_terminal_size()
    chart_height = term.lines - _status_height(ctx)
    _draw_chart(ctx, width=term.columns, height=max(chart_height, 10))
    _draw_status(ctx)


def _place_legend(sub_df, x_order, x_indices, y_axis, z_dom, colors, args, estimator):
    """Place a manual legend, choosing upper-left or upper-right to avoid tall bars."""
    if not len(z_dom):
        return
    n = len(x_order)
    half = max(1, n // 2)
    def _side_avg(xs):
        vals = sub_df[sub_df[args.x].isin(xs)][y_axis]
        return float(vals.median()) if len(vals) > 0 else 0.0
    left_avg = _side_avg(x_order[:half])
    right_avg = _side_avg(x_order[n - half:])
    use_right = left_avg > right_avg

    y_vals = sub_df[y_axis].dropna()
    y_max = float(y_vals.max()) if len(y_vals) else 1.0
    y_min = float(y_vals.min()) if len(y_vals) else 0.0
    y_step = max((y_max - y_min) * 0.07, abs(y_max) * 0.05, 0.01)

    x_pos = x_indices[-1] if use_right else x_indices[0]
    alignment = "right" if use_right else "left"
    for s, (z_val, color) in enumerate(zip(z_dom, colors)):
        label = f"██ {z_val}"
        ptx.text(label, x_pos, y_max - s * y_step, color=color, alignment=alignment)


def _draw_chart(ctx, width=None, height=None):
    args = ctx["args"]
    df = ctx["df"]
    y_axis = ctx["y_axis"]
    z_dom = ctx["z_dom"]
    colors = ctx["colors"]

    config = get_current_config(ctx)
    sub_df = get_projection(df, config)

    if args.x_norm:
        sub_df = group_normalization("x", df, config, args, y_axis)
    elif args.z_norm:
        sub_df = group_normalization("z", df, config, args, y_axis)
    elif args.ref_norm:
        sub_df = ref_normalization(df, config, args, y_axis)

    if args.geomean:
        gm_df = sub_df.copy()
        gm_df[args.x] = "geomean"
        sub_df = pd.concat([sub_df, gm_df])

    estimator = scipy.stats.gmean if args.geomean else np.median
    x_order = sub_df[args.x].unique()
    x_labels = [str(v) for v in x_order]

    ptx.clear_figure()
    ptx.theme("dark")
    if ctx.get("bg"):
        canvas, axes, ticks = ctx["bg"]
        ptx.canvas_color(canvas)
        ptx.axes_color(axes)
        ptx.ticks_color(ticks)
    if width is not None and height is not None:
        ptx.plotsize(width, height)

    # plotext.plot() rejects string x-values (tries to parse them as dates).
    # Use 1-based integer indices and set explicit x-tick labels instead.
    x_indices = list(range(1, len(x_labels) + 1))

    if args.lines:
        for i, z_val in enumerate(z_dom):
            group = sub_df[sub_df[args.z] == z_val]
            y_at_x = (
                group.groupby(args.x)[y_axis]
                .apply(estimator)
                .reindex(x_order)
            )
            ptx.plot(x_indices, y_at_x.tolist(), color=colors[i])
        ptx.xticks(x_indices, x_labels)
    else:
        all_ys = []
        for z_val in z_dom:
            group = sub_df[sub_df[args.z] == z_val]
            y_at_x = (
                group.groupby(args.x)[y_axis]
                .apply(estimator)
                .reindex(x_order)
            )
            all_ys.append(y_at_x.fillna(0).tolist())

        if len(z_dom) == 1:
            ptx.bar(x_labels, all_ys[0], color=colors[0])
        else:
            ptx.multiple_bar(x_labels, all_ys, color=colors)

    ptx.xlabel(str(args.x))

    if args.x_norm or args.z_norm or args.ref_norm:
        suffix = "gain" if args.norm_reverse else "normalized"
        ptx.ylabel(f"{y_axis} ({suffix})")
    elif args.unit:
        ptx.ylabel(f"{y_axis} [{args.unit}]")
    else:
        ptx.ylabel(str(y_axis))

    title_parts = []
    for i, y in enumerate(args.y, start=1):
        marker = f"[{i}]" if y == y_axis else f"({i})"
        title_parts.append(f"{marker} {y}")
    ptx.title(" | ".join(title_parts))

    _place_legend(sub_df, x_order, x_indices, y_axis, z_dom, colors, args, estimator)

    if not ctx.get("_build"):
        _annotate(args, sub_df, x_order, estimator, y_axis, z_dom)

    if ctx.get("_build"):
        return ptx.build(), sub_df, x_order, estimator
    ptx.show()


def _annotate(args, sub_df, x_order, estimator, y_axis, z_dom):
    """Place horizontal value labels just above bars or line points."""
    if not (args.annotate or args.annotate_max or args.annotate_min):
        return

    fmt = f"{{:.{args.digits}f}}"
    # plotext assigns 1-based integer positions to string x-tick labels
    x_pos = {x_val: i + 1 for i, x_val in enumerate(x_order)}
    # small upward nudge so the label clears the bar top (3% of y range)
    y_max = sub_df[y_axis].max()
    y_min = sub_df[y_axis].min()
    nudge = (y_max - y_min) * 0.03 + y_max * 0.01

    # For multiple_bar, plotext places bar i (0-indexed among N bars) at:
    #   group_center + (i - (N-1)/2) * (bar_group_width / N)
    # where bar_group_width defaults to 0.8.
    n = len(z_dom)
    bar_group_width = 0.8
    z_x_offset = {
        z_val: (i - (n - 1) / 2) * (bar_group_width / n)
        for i, z_val in enumerate(z_dom)
    }

    for z_val in z_dom:
        group = sub_df[sub_df[args.z] == z_val]
        y_at_x = (
            group.groupby(args.x)[y_axis]
            .apply(estimator)
            .reindex(x_order)
            .dropna()
        )

        if args.annotate_max:
            items = [y_at_x.idxmax()] if not y_at_x.empty else []
        elif args.annotate_min:
            items = [y_at_x.idxmin()] if not y_at_x.empty else []
        else:
            items = list(y_at_x.index)

        for x_val in items:
            y_val = y_at_x[x_val]
            x = x_pos[x_val] + z_x_offset[z_val]
            ptx.text(fmt.format(y_val), x, y_val + nudge)


def _draw_status(ctx):
    free_dims = ctx["free_dims"]
    domains = ctx["domains"]
    position = ctx["position"]
    selected_index = ctx["selected_index"]
    y_dims = ctx["y_dims"]
    y_axis = ctx["y_axis"]
    lock_dims = ctx["lock_dims"]

    lines = []

    # y-axis selector row (only when multiple y metrics exist)
    if len(y_dims) > 1:
        y_parts = []
        for i, y in enumerate(y_dims, start=1):
            if y == y_axis:
                y_parts.append(f"{_BOLD}{i}:{y}{_RESET}")
            else:
                y_parts.append(f"{_DIM}{i}:{y}{_RESET}")
        lines.append("  ".join(y_parts))

    # free-dimension row
    if free_dims:
        dim_parts = []
        for i, dim in enumerate(free_dims):
            if dim in lock_dims:
                value = lock_dims[dim]
                label = f"{_DIM}{dim}={value}{_RESET}"
                arrows = "  "
            else:
                value = domains[dim][position[dim]]
                has_choices = domains[dim].size > 1
                if i == selected_index:
                    label = f"{_REVERSE}{dim}={value}{_RESET}"
                    arrows = "\u2191\u2193" if has_choices else "  "
                else:
                    label = f"{dim}={value}"
                    arrows = f"{_DIM}\u2191\u2193{_RESET}" if has_choices else "  "
            dim_parts.append(f"{label} {arrows}")
        lines.append("  |  ".join(dim_parts))

    hint = (
        f"{_DIM}"
        "[\u2190\u2192 switch dim  \u2191\u2193 cycle value  "
        + ("1-9 y-axis  " if len(y_dims) > 1 else "")
        + "c colors  C background  . save  q quit]"
        f"{_RESET}"
    )
    lines.append(hint)

    print()
    for line in lines:
        print(" ", line)


def _handle_key(key, ctx):
    """Process a keypress; return False to exit the loop."""
    free_dims = ctx["free_dims"]
    domains = ctx["domains"]
    position = ctx["position"]
    y_dims = ctx["y_dims"]
    selected_index = ctx["selected_index"]

    if key in (b"q", b"Q", b"\x03"):
        return False

    elif key == b"\x1b[A":  # up arrow — next value
        if selected_index is not None:
            dim = free_dims[selected_index]
            if dim not in ctx["lock_dims"]:
                position[dim] = (position[dim] + 1) % domains[dim].size

    elif key == b"\x1b[B":  # down arrow — previous value
        if selected_index is not None:
            dim = free_dims[selected_index]
            if dim not in ctx["lock_dims"]:
                position[dim] = (position[dim] - 1) % domains[dim].size

    elif key == b"\x1b[D":  # left arrow — previous dimension
        if selected_index is not None and free_dims:
            ctx["selected_index"] = (selected_index - 1) % len(free_dims)

    elif key == b"\x1b[C":  # right arrow — next dimension
        if selected_index is not None and free_dims:
            ctx["selected_index"] = (selected_index + 1) % len(free_dims)

    elif len(key) == 1 and chr(key[0]).isdigit() and chr(key[0]) != "0":
        new_idx = int(chr(key[0])) - 1
        if new_idx < len(y_dims):
            ctx["y_axis"] = y_dims[new_idx]
            compute_ylimits(ctx)

    elif key == b"c":
        ctx["colors"] = _new_colors(len(ctx["z_dom"]))
        _save_cache(ctx)

    elif key == b"C":
        ctx["bg"] = _new_bg()
        _save_cache(ctx)

    elif key == b".":
        _save(ctx)
        return True  # skip re-render after save (already shown INFO message)

    return True


def _save(ctx):
    outfile = get_config_name(ctx) + ".txt"
    try:
        ctx["_build"] = True
        try:
            term = os.get_terminal_size()
            width, chart_height = term.columns, term.lines - _status_height(ctx)
        except OSError:
            width, chart_height = 120, 35
        chart_height = max(chart_height, 10)
        text, sub_df, x_order, estimator = _draw_chart(ctx, width=width, height=chart_height)
        # Replace colored bar blocks with per-group ASCII chars before stripping ANSI.
        _fill_chars = ["#", "@", "*", "%", "o", "x", "=", "~", "&", "^"]
        for i, color in enumerate(ctx.get("colors", [])):
            r, g, b = color
            ansi_color = re.escape(f"\x1b[38;2;{r};{g};{b}m")
            fill = _fill_chars[i % len(_fill_chars)]
            text = re.sub(ansi_color + "(█+)", lambda m, f=fill: f * len(m.group(1)), text)
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        args = ctx["args"]
        y_axis = ctx["y_axis"]
        z_dom = ctx["z_dom"]
        if args.annotate or args.annotate_max or args.annotate_min:
            fmt = f"{{:.{args.digits}f}}"
            col_w = max(len(str(v)) for v in x_order) + 2
            val_w = args.digits + 4
            header = " " * (val_w + 2) + "".join(str(x).ljust(col_w) for x in x_order)
            text += "\n" + header + "\n"
            for z_val in z_dom:
                group = sub_df[sub_df[args.z] == z_val]
                y_at_x = (
                    group.groupby(args.x)[y_axis]
                    .apply(estimator)
                    .reindex(x_order)
                )
                if args.annotate_max:
                    display = {x: (fmt.format(y_at_x[x]) if x == y_at_x.idxmax() else "-") for x in x_order}
                elif args.annotate_min:
                    display = {x: (fmt.format(y_at_x[x]) if x == y_at_x.idxmin() else "-") for x in x_order}
                else:
                    display = {x: fmt.format(y_at_x[x]) if not pd.isna(y_at_x[x]) else "-" for x in x_order}
                row = str(z_val).ljust(val_w + 2) + "".join(display[x].ljust(col_w) for x in x_order)
                text += row + "\n"
        config = get_current_config(ctx)
        if config:
            footer = "  ".join(f"{k}={v}" for k, v in config.items())
            text += "\n" + footer + "\n"
        with open(outfile, "w") as f:
            f.write(text)
        report(LogLevel.INFO, f"saved to '{outfile}'")
    except Exception as e:
        report(LogLevel.ERROR, f"could not save figure: {e}")
    finally:
        ctx["_build"] = False


def start_interactive(ctx):
    n = len(ctx["z_dom"])
    cache = _load_cache()
    cached_colors = cache.get("colors")
    ctx["colors"] = cached_colors if cached_colors and len(cached_colors) == n else _new_colors(n)
    ctx["bg"] = cache.get("bg")
    _render(ctx)

    while True:
        try:
            key = _getch()
        except (KeyboardInterrupt, EOFError):
            break

        keep_going = _handle_key(key, ctx)
        if not keep_going:
            break

        _render(ctx)


def launch(args):
    ctx = {"args": args}
    validate_files(ctx)
    locate_files(ctx)
    generate_dataframe(ctx)
    combine_dimensions(ctx)
    generate_derived_metrics(ctx)
    explode_array_metrics(ctx)
    validate_args(ctx)
    reorder_and_numericize(ctx)
    rescale(ctx)
    generate_space(ctx)
    compute_ylimits(ctx)
    start_interactive(ctx)

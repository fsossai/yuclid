from yuclid.log import report, LogLevel
from yuclid.plot import (
    validate_files,
    locate_files,
    generate_dataframe,
    combine_dimensions,
    update_table,
    get_current_config,
    get_projection,
    get_palette,
    set_axes_style,
    _esc,
    on_close,
    reorder_and_numericize,
)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy.stats as scipy_stats
import pandas as pd
import numpy as np


_DISTRIBUTIONS = {
    "normal":      scipy_stats.norm,
    "lognormal":   scipy_stats.lognorm,
    "exponential": scipy_stats.expon,
    "gamma":       scipy_stats.gamma,
    "beta":        scipy_stats.beta,
}


def validate_args_stats(ctx):
    args = ctx["args"]
    df = ctx["df"]

    # auto-detect numeric metric columns (same logic as plot.validate_args)
    exclude = [args.z] if args.z else []
    numeric_cols = (
        df.drop(columns=[c for c in exclude if c in df.columns])
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

    if len(args.y) == 0:
        if len(numeric_cols) == 0:
            report(
                LogLevel.FATAL,
                "no numeric columns found in the data",
                hint="use -y to specify a metric column",
            )
        report(LogLevel.INFO, "using {} as metric".format(", ".join(numeric_cols)))
        args.y = numeric_cols
    else:
        for col in args.y:
            if col not in df.columns:
                report(
                    LogLevel.FATAL,
                    f"metric column '{col}' not found",
                    hint="available columns: {}".format(", ".join(df.columns)),
                )
        # drop unselected numeric columns so they don't appear as free dims
        drop_cols = [col for col in numeric_cols if col not in args.y]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    # auto-select z from non-metric columns
    if args.z is None:
        candidates = df.columns.difference(args.y)
        if len(candidates) == 0:
            report(LogLevel.FATAL, "no columns available for grouping (-z)")
        args.z = candidates[np.argmin([df[c].nunique() for c in candidates])]
        report(LogLevel.INFO, f"using '{args.z}' as grouping dimension (-z)")
    elif args.z not in df.columns:
        report(
            LogLevel.FATAL,
            f"grouping column '{args.z}' not found",
            hint="available columns: {}".format(", ".join(df.columns.difference(args.y))),
        )

    if args.z in args.y:
        report(LogLevel.FATAL, "-z must be different from the metric columns")

    # lock_dims
    if len(args.lock_dims) > 0:
        pairs = {}
        for item in args.lock_dims:
            if "=" not in item:
                report(LogLevel.FATAL, f"invalid --lock-dims argument '{item}', expected key=value")
            k, v = item.split("=", 1)
            pairs[k] = v
        free = df.columns.difference(args.y + [args.z])
        for k in pairs:
            if k not in df.columns:
                report(
                    LogLevel.FATAL,
                    f"invalid lock dimension '{k}'",
                    hint=f"available: {', '.join(df.columns)}",
                )
            if k not in free:
                report(
                    LogLevel.FATAL,
                    f"cannot lock '{k}': it is not a free dimension",
                    hint=f"free dimensions: {', '.join(free)}",
                )
        ctx["lock_dims"] = pairs
    else:
        ctx["lock_dims"] = {}

    # warn if any z-group has very few observations
    for z_val in df[args.z].unique():
        n = len(df[df[args.z] == z_val][args.y[0]].dropna())
        if n < 5:
            report(
                LogLevel.WARNING,
                f"z-group '{z_val}' has only {n} observations — distribution fit may be unreliable",
            )

    ctx["y_dims"] = args.y
    ctx["y_axis"] = args.y[0]


def generate_space_stats(ctx):
    args, df = ctx["args"], ctx["df"]
    lock_keys = list(ctx["lock_dims"].keys())
    free_dims = list(df.columns.difference(args.y + [args.z] + lock_keys))
    domains = {d: df[d].unique() for d in df.columns}
    position = {d: 0 for d in df.columns}
    ctx["free_dims"] = free_dims
    ctx["selected_index"] = 0 if free_dims else None
    ctx["domains"] = domains
    ctx["position"] = position
    ctx["z_dom"] = df[args.z].unique()
    ctx["z_size"] = df[args.z].nunique()


def initialize_figure_stats(ctx):
    fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [20, 1]})
    ctx["fig"] = fig
    ctx["ax_plot"] = axs[0]
    ctx["ax_table"] = axs[1]
    axs[0].grid(axis="x")
    set_axes_style(ctx)
    y = axs[1].get_position().y1 + 0.03
    fig.add_artist(
        mlines.Line2D(
            [0.05, 0.95], [y, y],
            linewidth=4,
            transform=fig.transFigure,
            color="lightgrey",
        )
    )
    fig.subplots_adjust(top=0.92, bottom=0.1, hspace=0.3)
    fig.canvas.mpl_connect("key_press_event", lambda e: on_key_stats(e, ctx))
    fig.canvas.mpl_connect("close_event", lambda e: on_close(e, ctx))


def update_plot_stats(ctx):
    args = ctx["args"]
    df = ctx["df"]
    ax = ctx["ax_plot"]
    y_axis = ctx["y_axis"]
    z_dom = ctx["z_dom"]

    config = get_current_config(ctx)
    sub_df = get_projection(df, config)
    palette = get_palette(z_dom, colorblind=args.colorblind)

    ax.clear()
    ctx["fig"].suptitle(rf"Distribution of $\mathbf{{{_esc(y_axis)}}}$")

    for z_val in z_dom:
        data = sub_df[sub_df[args.z] == z_val][y_axis].dropna().values
        color = palette[z_val]

        ax.hist(
            data,
            bins=args.bins or "auto",
            density=True,
            alpha=0.4,
            color=color,
            label=str(z_val),
            edgecolor="white",
            linewidth=0.5,
        )

        if args.mean:
            ax.axvline(np.mean(data), color="red", lw=1.5, linestyle="--", zorder=3)
        if args.median:
            ax.axvline(np.median(data), color=color, lw=1.5, linestyle="-", zorder=3)

        if args.distribution != "none" and len(data) >= 2:
            dist = _DISTRIBUTIONS[args.distribution]
            try:
                params = dist.fit(data)
                x_range = np.linspace(data.min(), data.max(), 300)
                ax.plot(x_range, dist.pdf(x_range, *params), color=color, lw=2.5)
            except Exception as e:
                report(
                    LogLevel.WARNING,
                    f"could not fit {args.distribution} for z={z_val}: {e}",
                )

    unit_label = f" [{args.unit}]" if args.unit else ""
    ax.set_xlabel(rf"$\mathbf{{{_esc(y_axis)}}}${unit_label}")
    ax.set_ylabel("")
    ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    if args.mean:
        handles.append(mlines.Line2D([], [], color="red", lw=1.5, linestyle="--", label="mean"))
    if args.median:
        handles.append(mlines.Line2D([], [], color="grey", lw=1.5, linestyle="-", label="median"))
    ax.legend(handles=handles, title=_esc(args.z))
    ctx["fig"].canvas.draw_idle()


def _save_stats(ctx):
    y_axis = ctx["y_axis"]
    outfile = f"{y_axis}_distribution.pdf"
    ctx["fig"].savefig(outfile, bbox_inches="tight")
    report(LogLevel.INFO, f"saved to '{outfile}'")


def on_key_stats(event, ctx):
    selected_index = ctx["selected_index"]
    free_dims = ctx["free_dims"]
    domains = ctx["domains"]
    position = ctx["position"]

    if event.key in ["enter", " ", "up", "down"]:
        if selected_index is None:
            return
        selected_dim = free_dims[selected_index]
        if selected_dim in ctx["lock_dims"]:
            return
        x = 1 if event.key in [" ", "enter", "up"] else -1
        position[selected_dim] = (position[selected_dim] + x) % domains[selected_dim].size
        update_plot_stats(ctx)
        update_table(ctx)
    elif event.key in ["left", "right"]:
        if selected_index is None or len(free_dims) == 0:
            return
        delta = 1 if event.key == "right" else -1
        ctx["selected_index"] = (selected_index + delta) % len(free_dims)
        update_table(ctx)
    elif event.key in "123456789":
        new_idx = int(event.key) - 1
        if new_idx < len(ctx["y_dims"]):
            ctx["y_axis"] = ctx["y_dims"][new_idx]
            update_plot_stats(ctx)
    elif event.key == ".":
        _save_stats(ctx)


def launch(args):
    ctx = {"args": args}
    validate_files(ctx)
    locate_files(ctx)
    generate_dataframe(ctx)
    combine_dimensions(ctx)
    validate_args_stats(ctx)
    reorder_and_numericize(ctx)
    generate_space_stats(ctx)
    initialize_figure_stats(ctx)
    update_plot_stats(ctx)
    update_table(ctx)
    report(LogLevel.INFO, "application running")
    plt.show()

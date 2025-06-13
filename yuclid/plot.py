from yuclid.log import report, LogLevel
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import yuclid.spread as spread
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats
import subprocess
import threading
import itertools
import pathlib
import hashlib
import time
import math
import sys


def get_current_config(ctx):
    df = ctx["df"]
    domains = ctx["domains"]
    position = ctx["position"]
    free_dims = ctx["free_dims"]
    config = dict()
    for d in free_dims:
        k = domains[d][position[d]]
        config[d] = k
    return config


def get_config(point, keys):
    config = dict()
    for i, k in enumerate(keys):
        if i < len(point):
            config[k] = point[i]
        else:
            config[k] = None
    return config


def get_projection(df, config):
    keys = list(config.keys())
    if len(keys) == 0:
        return df
    mask = (df[keys] == pd.Series(config)).all(axis=1)
    return df[mask].copy()


def group_normalization(df, config, args, y_axis):
    sub_df = get_projection(df, config)
    ref_config = {k: v for k, v in config.items()} # copy
    selector = dict(pair.split("=") for pair in args.group_norm)
    ref_config.update(selector)

    # fixing types
    for k, v in ref_config.items():
        ref_config[k] = df[k].dtype.type(v)
        
    ref_df = get_projection(df, ref_config)
    estimator = scipy.stats.gmean if args.geomean else np.median
    gb_cols = df.columns.difference(args.y).tolist()
    ref = ref_df.groupby(gb_cols)[y_axis].apply(estimator).reset_index()
    sub_df[y_axis] /= sub_df[args.x].map(
        lambda x: ref[ref[args.x] == x][y_axis].values[0]
    )
    return sub_df


def validate_files(args):
    valid_files = []
    valid_formats = [".json", ".csv"]
    for file in args.files:
        if pathlib.Path(file).suffix in valid_formats:
            valid_files.append(file)
        else:
            report(LogLevel.ERROR, f"unsupported file format {file}")
    return valid_files


def get_local_mirror(rfile):
    return pathlib.Path(rfile.split(":")[1]).name


def locate_files(valid_files):
    local_files = []
    for file in valid_files:
        if is_remote(file):
            local_files.append(get_local_mirror(file))
        else:
            local_files.append(file)
    return local_files


def initialize_figure(ctx):
    fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [20, 1]})
    fig.set_size_inches(12, 10)
    ax_plot = axs[0]
    ax_table = axs[1]
    sns.set_theme(style="whitegrid")
    ax_plot.grid(axis="y")
    y = ax_table.get_position().y1 + 0.03
    line = mlines.Line2D(
        [0.05, 0.95], [y, y], linewidth=4, transform=fig.transFigure, color="lightgrey"
    )
    fig.add_artist(line)
    fig.subplots_adjust(top=0.92, bottom=0.1, hspace=0.3)
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, ctx))
    fig.canvas.mpl_connect("close_event", lambda event: on_close(event, ctx))
    ctx["fig"] = fig
    ctx["ax_plot"] = ax_plot
    ctx["ax_table"] = ax_table


def generate_dataframe(ctx):
    args = ctx["args"]
    local_files = ctx["local_files"]
    dfs = dict()
    for file in local_files:
        file = pathlib.Path(file)
        try:
            if file.suffix == ".json":
                dfs[file.stem] = pd.read_json(file, lines=True)
            elif file.suffix == ".csv":
                dfs[file.stem] = pd.read_csv(file)
        except:
            report(LogLevel.ERROR, f"could not open {file}")

    if len(dfs) == 0:
        report(LogLevel.ERROR, "no valid source of data")
        ctx["alive"] = False
        sys.exit(1)

    df = pd.concat(dfs)
    df = df.reset_index(level=0, names=["file"])

    if args.filter is None:
        user_filter = dict()
    else:
        user_filter = dict(pair.split("=") for pair in args.filter)
    for k, v in user_filter.items():
        user_filter[k] = df[k].dtype.type(v)

    if user_filter:
        user_filter_mask = (df[list(user_filter.keys())] == user_filter.values()).all(
            axis=1
        )
        df = df[user_filter_mask]

    ctx["df"] = df


def rescale(ctx):
    df = ctx["df"]
    args = ctx["args"]
    for y in args.y:
        df[y] = df[y] * args.rescale
    ctx["df"] = df


def generate_space(ctx):
    args = ctx["args"]
    df = ctx["df"]
    z_size = df[args.z].nunique()
    free_dims = list(df.columns.difference([args.x, args.z] + args.y))
    selected_index = 0 if len(free_dims) > 0 else None
    domains = dict()
    position = dict()

    for d in df.columns:
        domains[d] = df[d].unique()
        position[d] = 0

    z_dom = df[args.z].unique()
    ctx.update(
        {
            "z_size": z_size,
            "free_dims": free_dims,
            "selected_index": selected_index,
            "domains": domains,
            "position": position,
            "z_dom": z_dom,
        }
    )


def file_monitor(ctx):
    current_hash = None
    last_hash = None
    while ctx["alive"]:
        try:
            current_hash = ""
            for file in ctx["local_files"]:
                with open(file, "rb") as f:
                    current_hash += hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            current_hash = None
        if current_hash != last_hash:
            generate_dataframe(ctx)
            rescale(ctx)
            generate_space(ctx)
            compute_ylimits(ctx)
            space_columns = ctx["df"].columns.difference([ctx["y_axis"]])
            sizes = ["{}={}".format(d, ctx["df"][d].nunique()) for d in space_columns]
            missing = compute_missing(ctx)
            report(LogLevel.INFO, "space sizes", " | ".join(sizes))
            if len(missing) > 0:
                report(LogLevel.WARNING, f"at least {len(missing)} missing experiments")
            update_table(ctx)
            update_plot(ctx)
        last_hash = current_hash
        time.sleep(1)


def update_table(ctx):
    ax_table = ctx["ax_table"]
    free_dims = ctx["free_dims"]
    domains = ctx["domains"]
    position = ctx["position"]
    selected_index = ctx["selected_index"]
    ax_table.clear()
    ax_table.axis("off")
    if len(free_dims) == 0:
        return
    arrow_up = "\u2191"
    arrow_down = "\u2193"
    fields = []
    values = []
    arrows = []
    for i, d in enumerate(free_dims, start=1):
        value = domains[d][position[d]]
        if d == free_dims[selected_index]:
            fields.append(rf"$\mathbf{{{d}}}$")
            values.append(f"{value}")
            arrows.append(f"{arrow_up}{arrow_down}")
        else:
            fields.append(rf"$\mathbf{{{d}}}$")
            values.append(value)
            arrows.append("")
    ax_table.table(
        cellText=[fields, values, arrows], cellLoc="center", edges="open", loc="center"
    )
    ctx["fig"].canvas.draw_idle()


def is_remote(file):
    return "@" in file


def sync_files(ctx):
    args = ctx["args"]
    valid_files = ctx["valid_files"]
    jobs = []
    for file in valid_files:
        if is_remote(file):
            mirror = get_local_mirror(file)
            proc = subprocess.run(["scp", file, mirror])
            if proc.returncode != 0:
                report(LogLevel.ERROR, f"scp transfer failed for {file}")
                sys.exit(1)
            jobs.append((file, mirror))

    def rsync(src, dst):
        while ctx["alive"]:
            subprocess.run(
                ["rsync", "-z", "--checksum", src, dst],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(args.rsync_interval)

    for job in jobs:
        threading.Thread(target=rsync, daemon=True, args=job).start()


def fontsize_to_y_units(ctx, fontsize):
    fig = ctx["fig"]
    ax = ctx["ax_plot"]
    dpi = fig.dpi
    font_px = fontsize * dpi / 72
    inv = ax.transData.inverted()
    _, y0 = inv.transform((0, 0))
    _, y1 = inv.transform((0, font_px))
    dy = y1 - y0
    return dy


def autospace_annotations(ctx, fontsize, ys, padding_factor=1.10):
    text_height = fontsize_to_y_units(ctx, fontsize)
    h = text_height * padding_factor
    x_domain = ctx["domains"][ctx["args"].x]

    for x in x_domain:
        y_vals = [(z, ys[z][x]) for z in ys]
        lower_bound = -float("inf")
        for z, y in sorted(y_vals, key=lambda item: item[1]):
            box_bottom, box_top = y - h / 2, y + h / 2
            if box_bottom < lower_bound:  # overlap?
                shift = lower_bound - box_bottom
                new_y = y + shift
                lower_bound += box_top + shift
            else:
                lower_bound = box_top
                new_y = y
            ys[z][x] = new_y


def annotate(ctx, plot_type, sub_df, y_axis, palette):
    args = ctx["args"]
    ax_plot = ctx["ax_plot"]

    if not (args.annotate_max or args.annotate_min or args.annotate):
        return

    annotation_kwargs = {
        "ha": "center",
        "va": "bottom",
        "color": "black",
        "fontsize": 12,
        "fontweight": "normal",
        "xytext": (0, 5),
        "textcoords": "offset points",
    }

    ys = dict()
    z_domain = sub_df[args.z].unique()
    x_domain = sub_df[args.x].unique()

    for z in z_domain:
        group = sub_df[sub_df[args.z] == z]
        ys_z = group.groupby(args.x)[y_axis].apply(
            scipy.stats.gmean if args.geomean else np.median
        )
        ys[z] = ys_z

    autospace_annotations(ctx, annotation_kwargs["fontsize"], ys)

    x_adjust = {z: dict() for z in z_domain}

    # adjust x positions for annotations based on the plot type
    if plot_type == "lines":
        for z in z_domain:
            for x in x_domain:
                x_adjust[z][x] = x  # no adjustment needed for lines
    elif plot_type == "bars":

        def x_flat_generator():
            for p in ax_plot.patches:
                height = p.get_height()
                if not np.isnan(height) and height > 0:
                    yield p.get_x() + p.get_width() / 2

        x_flat_gen = iter(x_flat_generator())
        for z in z_domain:
            for x in x_domain:
                x_adjust[z][x] = next(x_flat_gen)

    for z in z_domain:
        annotation_kwargs_z = annotation_kwargs.copy()
        annotation_kwargs_z["color"] = palette[z]
        if args.annotate_max:
            y = ys[z].max()
            x = ys[z].idxmax()
            x = x_adjust[z][x]
            ax_plot.annotate(
                f"{y:.2f}",
                (x, y),
                **annotation_kwargs_z,
            )
        if args.annotate_min:
            y = ys[z].min()
            x = ys[z].idxmin()
            x = x_adjust[z][x]
            ax_plot.annotate(
                f"{y:.2f}",
                (x, y),
                **annotation_kwargs_z,
            )
        if args.annotate:
            for x, y in ys[z].items():
                x = x_adjust[z][x]
                ax_plot.annotate(
                    f"{y:.2f}",
                    (x, y),
                    **annotation_kwargs_z,
                )


def update_plot(ctx, padding_factor=1.05):
    args = ctx["args"]
    df = ctx["df"]
    free_dims = ctx["free_dims"]
    domains = ctx["domains"]
    position = ctx["position"]
    y_axis = ctx["y_axis"]
    z_dom = ctx["z_dom"]
    z_size = ctx["z_size"]
    ax_plot = ctx["ax_plot"]
    top = ctx.get("top", None)

    config = get_current_config(ctx)
    sub_df = get_projection(df, config)

    ax_plot.clear()

    initial_sub_df = sub_df.copy()

    def to_engineering_si(x, precision=0, unit=None):
        if x == 0:
            return f"{0:.{precision}f}"
        si_prefixes = {
            -24: "y",
            -21: "z",
            -18: "a",
            -15: "f",
            -12: "p",
            -9: "n",
            -6: "µ",
            -3: "m",
            0: "",
            3: "k",
            6: "M",
            9: "G",
            12: "T",
            15: "P",
            18: "E",
            21: "Z",
            24: "Y",
        }
        exp = int(math.floor(math.log10(abs(x)) // 3 * 3))
        exp = max(min(exp, 24), -24)  # clamp to available prefixes
        coeff = x / (10**exp)
        prefix = si_prefixes.get(exp, f"e{exp:+03d}")
        unit = unit or ""
        return f"{coeff:.{precision}f}{prefix}{unit}"

    if args.group_norm is not None:
        sub_df = group_normalization(df, config, args, y_axis)
        if args.geomean:
            gm_df = sub_df.copy()
            gm_df[args.x] = "geomean"
            sub_df = pd.concat([sub_df, gm_df])

    if args.ref_norm is not None:
        c1 = sub_df[args.z] == args.ref_norm
        c2 = sub_df[args.x] == sub_df[args.x].min()
        baseline = sub_df[(c1 & c2)][y_axis].min()
        sub_df[y_axis] = baseline / sub_df[y_axis]

    if args.group_norm is not None or args.ref_norm is not None:
        ax_plot.axhline(y=1.0, linestyle="-", linewidth=4, color="lightgrey")

    def custom_error(data):
        d = pd.DataFrame(data)
        return (
            spread.lower(args.spread_measure)(d),
            spread.upper(args.spread_measure)(d),
        )

    if args.colorblind:
        palette = sns.color_palette("colorblind", n_colors=len(z_dom))
        palette = {z: palette[i] for i, z in enumerate(z_dom)}
    else:
        preferred_colors = [
            "#5588dd",
            "#882255",
            "#33bb88",
            "#ddcc77",
            "#cc6677",
            "#999933",
            "#aa44ff",
            "#448811",
            "#3fa7d6",
            "#e94f37",
            "#6cc551",
            "#dabef9",
        ]
        color_gen = iter(preferred_colors)
        palette = {z: next(color_gen) for z in z_dom}

    if args.lines:
        sns.lineplot(
            data=sub_df,
            x=args.x,
            y=y_axis,
            hue=args.z,
            palette=palette,
            lw=2,
            linestyle="-",
            marker="o",
            errorbar=None,
            ax=ax_plot,
            estimator=np.median,
        )
        spread.draw(
            ax_plot,
            [args.spread_measure],
            sub_df,
            x=args.x,
            y=y_axis,
            z=args.z,
            palette=palette,
        )
        annotate(ctx, "lines", sub_df, y_axis, palette)
    else:
        sns.barplot(
            data=sub_df,
            ax=ax_plot,
            estimator=scipy.stats.gmean if args.geomean else np.median,
            palette=palette,
            legend=True,
            x=args.x,
            y=y_axis,
            hue=args.z,
            errorbar=custom_error,
            alpha=0.6,
            err_kws={
                "color": "black",
                "alpha": 1.0,
                "linewidth": 2.0,
                "solid_capstyle": "round",
                "solid_joinstyle": "round",
            },
        )
        annotate(ctx, "bars", sub_df, y_axis, palette)

    def format_ylabel(label):
        if args.unit is None:
            return label
        elif args.ref_norm is None:
            return f"{label} [{args.unit}]"
        else:
            return f"{label}"

    if top is not None:
        ax_plot.set_ylim(top=top * padding_factor, bottom=0.0)
    if args.group_norm is not None:
        normalized_label = f"{y_axis} (normalized to {args.group_norm})"
        ax_plot.set_ylabel(format_ylabel(normalized_label))
    elif args.ref_norm is not None:
        refnorm_label = f"refnorm vs {args.ref_norm}"
        ax_plot.set_ylabel(format_ylabel(refnorm_label))
        handles, labels = ax_plot.get_legend_handles_labels()
        new_labels = [
            f"{args.ref_norm} (baseline)" if label == args.ref_norm else label
            for label in labels
        ]
        ax_plot.legend(handles, new_labels, loc="upper left")
    else:
        ax_plot.set_ylabel(format_ylabel(y_axis))

    # set figure title
    y_left, y_right = initial_sub_df[y_axis].min(), initial_sub_df[y_axis].max()
    y_range = "[{} - {}]".format(
        to_engineering_si(y_left, unit=args.unit),
        to_engineering_si(y_right, unit=args.unit),
    )
    title_parts = []
    for i, y in enumerate(args.y, start=1):
        if y == y_axis:
            title_parts.append(rf"{i}: $\mathbf{{{y}}}$")
        else:
            title_parts.append(f"{i}: {y}")
    title = " | ".join(title_parts) + "\n" + y_range
    ctx["fig"].suptitle(title)

    if args.geomean:
        pp = sorted(ax_plot.patches, key=lambda x: x.get_x())
        x = pp[-z_size].get_x() + pp[-z_size - 1].get_x() + pp[-z_size - 1].get_width()
        plt.axvline(x=x / 2, color="grey", linewidth=1, linestyle="-")

    ctx["fig"].canvas.draw_idle()


def get_config_name(ctx):
    y_axis = ctx["y_axis"]
    dims = ctx["free_dims"]
    domains = ctx["domains"]
    position = ctx["position"]
    status = ["speedup" if ctx["args"].ref_norm else y_axis]
    status += [str(domains[d][position[d]]) for d in dims]
    name = "_".join(status)
    return name


def get_status_description(ctx):
    args = ctx["args"]
    description_parts = []
    domains = ctx["domains"]

    for d in ctx["free_dims"]:
        position = ctx["position"]
        value = domains[d][position[d]]
        description_parts.append(f"{d}={value}")

    description = " | ".join(description_parts)
    if ctx["z_size"] == 1:
        z_values = ctx["df"][args.z].unique()
        description += f" | {args.z}={z_values[0]}"

    return description


def save_to_file(ctx, outfile=None):
    ax_plot = ctx["ax_plot"]
    outfile = outfile or get_config_name(ctx) + ".pdf"
    if ctx["z_size"] == 1:
        legend = ax_plot.get_legend()
        if legend:
            legend.set_visible(False)
    if ctx["args"].speedup:
        title_bold = "speedup"
    else:
        title_bold = str(ctx["y_axis"])

    title = rf"$\mathbf{{{title_bold}}}$" + "\n" + get_status_description(ctx)
    ctx["fig"].suptitle(title)
    extent = ax_plot.get_window_extent().transformed(
        ctx["fig"].dpi_scale_trans.inverted()
    )
    ctx["fig"].savefig(outfile, bbox_inches=extent.expanded(1.2, 1.2))
    report(LogLevel.INFO, f"saved to '{outfile}'")


def on_key(event, ctx):
    selected_index = ctx["selected_index"]
    free_dims = ctx["free_dims"]
    domains = ctx["domains"]
    position = ctx["position"]
    y_dims = ctx["y_dims"]

    if event.key in ["enter", " ", "up", "down"]:
        x = 1 if event.key in [" ", "enter", "up"] else -1
        if selected_index is None:
            return
        selected_dim = free_dims[selected_index]
        cur_pos = position[selected_dim]
        new_pos = (cur_pos + x) % domains[selected_dim].size
        position[selected_dim] = new_pos
        update_plot(ctx)
        update_table(ctx)
    elif event.key in ["left", "right"]:
        if selected_index is None:
            return
        if event.key == "left":
            ctx["selected_index"] = (selected_index - 1) % len(free_dims)
        else:
            ctx["selected_index"] = (selected_index + 1) % len(free_dims)
        update_table(ctx)
    elif event.key in "123456789":
        new_idx = int(event.key) - 1
        if new_idx < len(y_dims):
            ctx["y_axis"] = y_dims[new_idx]
            compute_ylimits(ctx)
            update_plot(ctx)
    elif event.key in ".":
        save_to_file(ctx)


def on_close(event, ctx):
    ctx["alive"] = False


def compute_missing(ctx):
    df = ctx["df"]
    y_dims = ctx["y_dims"]
    space_columns = df.columns.difference(y_dims)
    expected = set(itertools.product(*[df[col].unique() for col in space_columns]))
    observed = set(map(tuple, df[space_columns].drop_duplicates().values))
    missing = expected - observed
    return pd.DataFrame(list(missing), columns=space_columns)


def validate_dimensions(ctx, dims):
    args = ctx["args"]
    df = ctx["df"]
    for col in dims:
        if col not in df.columns:
            available = list(df.columns)
            hint = "available columns: {}".format(", ".join(available))
            report(LogLevel.FATAL, "invalid column", col, hint=hint)


def validate_args(ctx):
    args = ctx["args"]
    df = ctx["df"]

    # Y-axis
    numeric_cols = (
        df.drop(columns=[args.x])
        .select_dtypes(include=[np.float64, np.float32])
        .columns.tolist()
    )
    if args.y is None:
        # find the floating point numeric columns
        if len(numeric_cols) == 0:
            report(
                LogLevel.FATAL,
                "No numeric columns found in the data",
                hint="use -y to specify a Y-axis",
            )
        report(LogLevel.INFO, "Using '{}' as Y-axis".format(", ".join(numeric_cols)))
        args.y = numeric_cols
    validate_dimensions(ctx, args.y)
    for y in args.y:
        if not pd.api.types.is_numeric_dtype(df[y]):
            t = df[y].dtype
            if len(numeric_cols) > 0:
                hint = "try {}".format(
                    numeric_cols[0]
                    if len(numeric_cols) == 1
                    else ", ".join(numeric_cols)
                )
            else:
                hint = "use -y to specify a Y-axis"
            report(
                LogLevel.FATAL,
                f"Y-axis must have a numeric type. '{y}' has type '{t}'",
                hint=hint,
            )

    # X-axis
    validate_dimensions(ctx, [args.x])
    if args.x in args.y:
        report(
            LogLevel.FATAL,
            f"X-axis and Y-axis must be different dimensions",
        )

    # Z-axis
    # check that there are at least two dimensions other than args.y
    if len(df.columns.difference(args.y)) < 2:
        report(
            LogLevel.FATAL,
            "there must be at least two dimensions other than the Y-axis",
        )
    if args.z is None:
        # pick the first column that is not args.x or in args.y
        available = df.columns.difference([args.x] + args.y)
        args.z = available[np.argmin([df[col].nunique() for col in available])]
        report(LogLevel.INFO, "Using '{}' as Z-axis".format(args.z))
    else:
        validate_dimensions(ctx, [args.z])
    zdom = df[args.z].unique()
    if len(zdom) == 1 and args.geomean:
        report(
            LogLevel.WARNING,
            "--geomean is superfluous because '{}' is the only value in the '{}' group".format(
                zdom[0], args.z
            ),
        )

    # all axis
    if args.x == args.z or args.z in args.y:
        report(
            LogLevel.FATAL,
            "the -z dimension must be different from the dimension used on the X or Y axis",
        )

    # geomean and lines
    if args.geomean and args.lines:
        report(LogLevel.FATAL, "--geomean and --lines cannot be used together")
    for d in df.columns.difference(args.y):
        n = df[d].nunique()
        if n > 20 and pd.api.types.is_numeric_dtype(df[d]):
            report(
                LogLevel.WARNING,
                f"'{d}' seems to have many ({n}) numeric values. Are you sure this is not supposed to be the Y-axis?",
            )

    # speedup and normalize
    if args.ref_norm is not None:
        if not pd.api.types.is_numeric_dtype(df[args.x]):
            report(
                LogLevel.FATAL,
                "--speedup only works when the X-axis has a numeric type",
            )
    if args.group_norm is not None and args.ref_norm is not None:
        report(
            LogLevel.FATAL,
            "--group-norm and --ref-norm cannot be used together",
        )

    if args.show_missing:
        missing = compute_missing(ctx)
        if len(missing) > 0:
            report(LogLevel.WARNING, "missing experiments:")
            report(LogLevel.WARNING, "\n" + missing.to_string(index=False))
            report(LogLevel.WARNING, "")

    ctx["y_dims"] = args.y
    ctx["y_axis"] = args.y[0]


def start_gui(ctx):
    ctx["alive"] = True

    update_plot(ctx)
    update_table(ctx)
    threading.Thread(target=file_monitor, daemon=True, args=(ctx,)).start()
    report(LogLevel.INFO, "application running")
    time.sleep(0.5)
    plt.show()


def compute_ylimits(ctx):
    args = ctx["args"]
    free_dims = ctx["free_dims"]
    df = ctx["df"]
    y_axis = ctx["y_axis"]
    domains = ctx["domains"]
    free_domains = {k: v for k, v in domains.items() if k in free_dims}
    top = None
    if len(free_dims) == 0:
        ctx["top"] = None
        return
    if args.group_norm:
        top = 0
        for point in itertools.product(*free_domains.values()):
            filt = (df[free_domains.keys()] == point).all(axis=1)
            config = get_config(point, free_domains.keys())
            df_config = group_normalization(df, config, args, y_axis)
            zx = df_config.groupby([args.z, args.x])[y_axis]
            t = zx.apply(spread.upper(args.spread_measure))
            top = max(top, t.max())
    elif args.ref_norm:
        top = 0
        max_speedup = 1.0
        for point in itertools.product(*free_domains.values()):
            filt = (df[free_domains.keys()] == point).all(axis=1)
            df_config = df[filt].copy()
            c1 = df_config[args.z] == args.ref_norm
            c2 = df_config[args.x] == df_config[args.x].min()
            baseline = df_config[(c1 & c2)][y_axis].min()
            best = df_config[y_axis].min()
            speedup = baseline / best
            max_speedup = max(max_speedup, speedup)
        top = max_speedup
    else:
        top = df[y_axis].max()
    ctx["top"] = top


def launch(args):
    ctx = {"args": args, "alive": True}
    ctx["valid_files"] = validate_files(args)
    ctx["local_files"] = locate_files(ctx["valid_files"])
    sync_files(ctx)
    generate_dataframe(ctx)
    validate_args(ctx)
    rescale(ctx)
    generate_space(ctx)
    compute_ylimits(ctx)
    initialize_figure(ctx)
    start_gui(ctx)

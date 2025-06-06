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


def normalize(input_df, args, y_axis):
    b = input_df[args.z].dtype.type(args.normalize)
    estimator = scipy.stats.gmean if args.geomean else np.median
    ref = input_df.groupby([args.x, args.z])[y_axis]
    ref = ref.apply(lambda x: estimator(x))
    input_df[y_axis] /= input_df[args.x].map(lambda x: ref[(x, b)])


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
    fig.subplots_adjust(top=0.95, bottom=0.1, hspace=0.3)
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


def generate_space(ctx):
    args = ctx["args"]
    df = ctx["df"]
    y_dims = ctx["y_dims"]
    z_size = df[args.z].nunique()
    dims = list(df.columns.difference([args.x, args.z] + y_dims))
    dim_keys = "123456789"[: len(dims)]
    selected_index = 0 if len(dims) > 0 else None
    domain = dict()
    position = dict()
    for d in dims:
        domain[d] = df[d].unique()
        position[d] = 0
    z_dom = df[args.z].unique()
    ctx.update(
        {
            "z_size": z_size,
            "dims": dims,
            "dim_keys": dim_keys,
            "selected_index": selected_index,
            "domain": domain,
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
            generate_space(ctx)
            compute_ylimits(ctx)
            space_columns = ctx["df"].columns.difference([ctx["y_axis"]])
            sizes = [f"{d}={ctx['df'][d].nunique()}" for d in space_columns]
            missing = compute_missing(ctx)
            report(LogLevel.INFO, f"new space: {' | '.join(sizes)}")
            if len(missing) > 0:
                report(LogLevel.WARNING, f"at least {len(missing)} missing experiments")
            update_table(ctx)
            update_plot(ctx)
        last_hash = current_hash
        time.sleep(1)


def update_table(ctx):
    ax_table = ctx["ax_table"]
    dims = ctx["dims"]
    domain = ctx["domain"]
    position = ctx["position"]
    selected_index = ctx["selected_index"]
    ax_table.clear()
    ax_table.axis("off")
    if len(dims) == 0:
        return
    arrow_up = "\u2191"
    arrow_down = "\u2193"
    fields = []
    values = []
    arrows = []
    for i, d in enumerate(dims, start=1):
        value = domain[d][position[d]]
        if d == dims[selected_index]:
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


def update_plot(ctx, padding_factor=1.05):
    args = ctx["args"]
    df = ctx["df"]
    dims = ctx["dims"]
    domain = ctx["domain"]
    position = ctx["position"]
    y_axis = ctx["y_axis"]
    y_dims = ctx["y_dims"]
    z_dom = ctx["z_dom"]
    z_size = ctx["z_size"]
    ax_plot = ctx["ax_plot"]
    top = ctx.get("top", None)

    sub_df = df.copy()
    for d in dims:
        k = domain[d][position[d]]
        sub_df = sub_df[sub_df[d] == k]
    ax_plot.clear()

    def to_engineering_si(x, precision=0):
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
        return f"{coeff:.{precision}f}{prefix}"

    y_left, y_right = sub_df[y_axis].min(), sub_df[y_axis].max()
    y_range = "[{} - {}]".format(to_engineering_si(y_left), to_engineering_si(y_right))

    if args.normalize is not None:
        if args.geomean:
            gm_df = sub_df.copy()
            gm_df[args.x] = "geomean"
            cols = gm_df.columns.difference([y_axis]).to_list()
            gm_df = gm_df.groupby(cols)[y_axis].apply(scipy.stats.gmean).reset_index()
            sub_df = pd.concat([sub_df, gm_df])
        normalize(sub_df, args, y_axis)

    if args.speedup is not None:
        c1 = sub_df[args.z] == args.speedup
        c2 = sub_df[args.x] == sub_df[args.x].min()
        baseline = sub_df[(c1 & c2)][y_axis].min()
        sub_df[y_axis] = baseline / sub_df[y_axis]

    if args.normalize is not None or args.speedup is not None:
        ax_plot.axhline(y=1.0, linestyle="-", linewidth=4, color="lightgrey")

    def custom_error(data):
        d = pd.DataFrame(data)
        return (
            spread.lower(args.spread_measure)(d),
            spread.upper(args.spread_measure)(d),
        )

    if args.colorblind:
        palette = "colorblind"
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
    else:
        sns.barplot(
            data=sub_df,
            ax=ax_plot,
            estimator=np.median,
            palette=palette,
            legend=True,
            x=args.x,
            y=y_axis,
            hue=args.z,
            errorbar=custom_error,
            alpha=0.6,
        )

    if top is not None:
        ax_plot.set_ylim(top=top * padding_factor, bottom=0.0)
    if args.normalize is not None:
        ax_plot.set_ylabel(
            "{} (normalized to {})\n{}".format(
                ax_plot.get_ylabel(), args.normalize, y_range
            )
        )
    elif args.speedup is not None:
        ax_plot.set_ylabel("speedup vs {}\n{}".format(args.speedup, y_range))
        handles, labels = ax_plot.get_legend_handles_labels()
        new_labels = [
            f"{args.speedup} (baseline)" if label == args.speedup else label
            for label in labels
        ]
        ax_plot.legend(handles, new_labels, loc="upper left")
    else:
        ax_plot.set_ylabel("{}\n{}".format(y_axis, y_range))

    title = []
    for i, y in enumerate(y_dims, start=1):
        if y == y_axis:
            title.append(rf"{i}: $\mathbf{{{y}}}$")
        else:
            title.append(f"{i}: {y}")
    ax_plot.set_title("  |  ".join(title))

    if args.geomean:
        pp = sorted(ax_plot.patches, key=lambda x: x.get_x())
        x = pp[-z_size].get_x() + pp[-z_size - 1].get_x() + pp[-z_size - 1].get_width()
        plt.axvline(x=x / 2, color="grey", linewidth=1, linestyle="-")

    ctx["fig"].canvas.draw_idle()


def get_config_name(ctx):
    y_axis = ctx["y_axis"]
    dims = ctx["dims"]
    domain = ctx["domain"]
    position = ctx["position"]
    status = ["speedup" if ctx["args"].speedup else y_axis]
    status += [str(domain[d][position[d]]) for d in dims]
    name = "_".join(status)
    return name


def get_status_description(ctx):
    args = ctx["args"]
    description_parts = []
    if args.speedup:
        description_main = "speedup"
    else:
        description_main = str(ctx["y_axis"])
    description_parts.append(rf"$\mathbf{{{description_main}}}$")

    for d in ctx["dims"]:
        position = ctx["position"]
        value = ctx["domain"][d][position[d]]
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
    ax_plot.set_title(get_status_description(ctx))
    extent = ax_plot.get_window_extent().transformed(
        ctx["fig"].dpi_scale_trans.inverted()
    )
    ctx["fig"].savefig(outfile, bbox_inches=extent.expanded(1.2, 1.1))
    report(LogLevel.INFO, f"saved to '{outfile}'")


def on_key(event, ctx):
    selected_index = ctx["selected_index"]
    dims = ctx["dims"]
    domain = ctx["domain"]
    position = ctx["position"]
    y_dims = ctx["y_dims"]
    y_axis = ctx["y_axis"]
    if event.key in ["enter", " ", "up", "down"]:
        x = 1 if event.key in [" ", "enter", "up"] else -1
        if selected_index is None:
            return
        selected_dim = dims[selected_index]
        cur_pos = position[selected_dim]
        new_pos = (cur_pos + x) % domain[selected_dim].size
        position[selected_dim] = new_pos
        update_plot(ctx)
        update_table(ctx)
    elif event.key in ["left", "right"]:
        if selected_index is None:
            return
        if event.key == "left":
            ctx["selected_index"] = (selected_index - 1) % len(dims)
        else:
            ctx["selected_index"] = (selected_index + 1) % len(dims)
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


def validate_options(ctx):
    args = ctx["args"]
    df = ctx["df"]
    c = 0
    c += 1 if args.normalize is not None else 0
    c += 1 if args.speedup is not None else 0
    if c > 1:
        report(LogLevel.FATAL, "specifiy only one among --normalize, --speedup")
    if c == 0:
        if args.geomean:
            report(
                LogLevel.FATAL,
                "--geomean can only be used together with --normalize",
            )
    if args.normalize is not None or args.speedup is not None:
        available = df[args.z].unique()
        val = df[args.z].dtype.type(args.normalize or args.speedup)
        if val not in available:
            report(
                LogLevel.FATAL,
                "--normalize and --speedup must be one of the following values:",
                available,
            )
    if args.speedup is not None:
        if not pd.api.types.is_numeric_dtype(df[args.x]):
            report(
                LogLevel.FATAL,
                "--speedup only works when the X-axis has a numeric type.",
            )
    # Y-axis
    y_dims = args.y.split(",")
    y_axis = y_dims[0]
    for col in [args.x, args.z] + y_dims:
        if col not in df.columns:
            available = list(df.columns)
            hint = "available columns: {}".format(", ".join(available))
            report(LogLevel.FATAL, "invalid column", col, hint=hint)
    for y in y_dims:
        if not pd.api.types.is_numeric_dtype(df[y]):
            t = df[y].dtype
            report(
                LogLevel.FATAL,
                f"Y-axis must have a numeric type. '{y_axis}' has type '{t}'",
            )

    zdom = df[args.z].unique()
    if len(zdom) == 1 and args.geomean:
        report(
            LogLevel.WARNING,
            f"--geomean is superfluous because '{zdom[0]}' is the only value in the '{args.z}' group",
        )
    if args.geomean and args.lines:
        report(LogLevel.FATAL, "--geomean and --lines cannot be used together")
    if args.x in y_dims:
        report(
            LogLevel.FATAL,
            f"X-axis and Y-axis must be different dimensions. Given {args.x}",
        )
    if args.x == args.z or args.z in y_dims:
        report(
            LogLevel.FATAL,
            "the -z dimension must be different from the dimension used on the X or Y axis",
        )
    space_columns = df.columns.difference(y_dims)
    for d in space_columns:
        n = df[d].nunique()
        if n > 20 and pd.api.types.is_numeric_dtype(df[d]):
            report(
                LogLevel.WARNING,
                f"'{d}' seems to have many ({n}) numeric values. Are you sure this is not supposed to be the Y-axis?",
            )

    if args.show_missing:
        missing = compute_missing(ctx)
        if len(missing) > 0:
            report(LogLevel.WARNING, "missing experiments:")
            report(LogLevel.WARNING, "\n" + missing.to_string(index=False))
            report(LogLevel.WARNING, "")

    ctx["y_dims"] = y_dims
    ctx["y_axis"] = y_dims[0]


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
    dims = ctx["dims"]
    df = ctx["df"]
    y_axis = ctx["y_axis"]
    domain = ctx["domain"]
    top = None
    if len(dims) == 0:
        ctx["top"] = None
        return
    if args.normalize:
        top = 0
        for config in itertools.product(*domain.values()):
            filt = (df[domain.keys()] == config).all(axis=1)
            df_filtered = df[filt].copy()
            normalize(df_filtered, args, y_axis)
            zx = df_filtered.groupby([args.z, args.x])[y_axis]
            t = zx.apply(spread.upper(args.spread_measure))
            top = max(top, t.max())
    elif args.speedup:
        c1 = df[args.z] == args.speedup
        c2 = df[args.x] == df[args.x].min()
        not_y = df.columns.difference([y_axis]).tolist()
        baseline = df[(c1 & c2)].groupby(not_y)[y_axis].min().max()
        top = baseline / df[y_axis].min()
    else:
        top = df[y_axis].max()
    ctx["top"] = top


def launch(args):
    ctx = {"args": args, "alive": True}
    ctx["valid_files"] = validate_files(args)
    ctx["local_files"] = locate_files(ctx["valid_files"])
    sync_files(ctx)
    generate_dataframe(ctx)
    validate_options(ctx)
    generate_space(ctx)
    compute_ylimits(ctx)
    initialize_figure(ctx)
    start_gui(ctx)

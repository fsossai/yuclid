from yuclid import __version__
import yuclid.spread
import yuclid.plot
import yuclid.tplot
import yuclid.run
import yuclid.log
import argparse


def get_parser():
    parser = argparse.ArgumentParser(prog="yuclid", description="Yuclid CLI tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run experiments and collect data")
    run_parser.add_argument(
        "-i",
        "--inputs",
        default=None,
        nargs="*",
        help="Specify one or more configuration files, in JSON or YAML. "
        "Defaults to whichever of yuclid.json, yuclid.yaml or yuclid.yml "
        "exists. Objects and lists will be joined",
    )
    run_parser.add_argument(
        "--order",
        default=[],
        nargs="*",
        help="List of dimensions to override the order of experiments",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="JSON output file path for the generated data",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory path where the generated data will be saved",
    )
    run_parser.add_argument(
        "--format",
        default=None,
        choices=["jsonl", "csv"],
        help="Format of the generated data. Defaults to the extension of "
        "--output or --resume, and to jsonl otherwise",
    )
    run_parser.add_argument(
        "--temp-dir",
        default=None,
        help="Directory where the trials' output will be saved. "
        "Defaults to the run's own directory under .yuclid/runs",
    )
    run_parser.add_argument(
        "-p",
        "--presets",
        nargs="*",
        default=[],
        help="Specify a list of preset names to run. "
        "A subspace will be created and executed for each preset specified, "
        "one at a time",
    )
    run_parser.add_argument(
        "-s",
        "--select",
        nargs="*",
        default=[],
        help="Select a list of name=csv_values pairs for each dimension. E.g. dim1=val1,val2 dim2=val3,val4",
    )
    run_parser.add_argument(
        "--workspace",
        default=None,
        metavar="DIR",
        help="Keep Yuclid's state in DIR instead of ./.yuclid: the runs it "
        "records, their logs and their captured output",
    )
    run_parser.add_argument(
        "--until",
        action="append",
        default=None,
        metavar="RULE",
        help="Repeat each point until this holds, instead of a fixed count. "
        "Either a duration (3s, 500ms, 2m) or a precision on a metric "
        "(time±5%%, or time+-5%%). May be given more than once, and the first "
        "rule satisfied ends the point",
    )
    run_parser.add_argument(
        "--min-runs",
        type=int,
        default=None,
        metavar="N",
        help="Never stop a point before N repetitions (default: 3 with --until)",
    )
    run_parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        metavar="N",
        help="Never repeat a point more than N times (default: 100 with --until)",
    )
    run_parser.add_argument(
        "--points",
        default=None,
        metavar="FILE",
        help="Run exactly the points this file names, instead of the product "
        "of the configured space. Each entry is a set of coordinates, e.g. "
        '{"size": ["*"], "impl": ["dot"]}, and the run covers all of them. '
        "Cannot be combined with --select or --presets",
    )
    run_parser.add_argument(
        "--fold",
        default=False,
        action="store_true",
        help="Stores values produced by a metric in an array",
    )
    run_parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Show experiment that would run",
    )
    run_parser.add_argument(
        "--parallel-trials",
        type=int,
        default=0,
        nargs="?",
        const=-1,
        metavar="N",
        help="Run trials in the subspace in parallel. "
        "Optionally specify the max number of parallel workers (default: number of CPUs)",
    )
    run_parser.add_argument(
        "-m",
        "--metrics",
        nargs="*",
        default=None,
        help="Specify a list of metrics to compute among the registered ones. "
        "If empty, all metrics will be computed.",
    )
    run_parser.add_argument(
        "--no-setup",
        "--skip-setup",
        dest="no_setup",
        default=False,
        action="store_true",
        help="Skip the setup phase and run trials directly",
    )
    run_parser.add_argument(
        "--compile",
        default=None,
        metavar="FILE",
        help="Write a shell script that reproduces this configuration instead "
        "of running it. Every point is unrolled, so the script needs neither "
        "yuclid nor the configuration to run",
    )
    run_parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Skip the points the file given to --output already holds, and "
        "append the others to it. Requires --output",
    )
    run_parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=None,
        metavar="N",
        help="Run each point configuration N times (default: 1). Cannot be "
        "combined with --until, which decides the count instead",
    )
    run_parser.add_argument(
        "--abort-on-error",
        default=False,
        action="store_true",
        help="Stop at the first point that fails. By default a failing trial "
        "or metric is reported, the point is marked failed, and the run "
        "carries on with the rest of the space",
    )
    run_parser.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="A name for this run, to tell it apart from the others in "
        "`yuclid runs`. Runs are still addressed by their id",
    )
    run_parser.add_argument(
        "--replay",
        default=None,
        metavar="ID",
        help="Run the points a previous run measured. Same as --points with "
        "that run's points; `yuclid replay` is the shorter way to ask for it",
    )
    run_parser.add_argument(
        "--continue-run",
        default=None,
        metavar="ID",
        # internal: how `yuclid finish` writes into the run it is finishing
        # instead of starting one of its own
        help=argparse.SUPPRESS,
    )

    # plot subcommand — GUI
    plot_parser = subparsers.add_parser("plot", help="Plot data in a GUI")
    _add_plot_args(plot_parser)

    # tplot subcommand — terminal version of plot
    tplot_parser = subparsers.add_parser(
        "tplot", help="Plot data interactively in the terminal"
    )
    _add_plot_args(tplot_parser)

    # describe subcommand
    describe_parser = subparsers.add_parser(
        "describe", help="Describe what a result file holds"
    )
    _add_describe_args(describe_parser)

    # runs subcommand
    runs_parser = subparsers.add_parser(
        "runs", help="List the runs recorded under .yuclid"
    )
    runs_parser.add_argument(
        "--workspace",
        default=None,
        metavar="DIR",
        help="Read the runs recorded in DIR instead of ./.yuclid",
    )
    runs_parser.add_argument(
        "-n",
        type=int,
        default=10,
        metavar="N",
        help="Show at most N runs (default: %(default)s)",
    )
    runs_parser.add_argument(
        "--last",
        default=False,
        action="store_true",
        help="Print only the results file of the most recent run, so that it "
        "can be passed to another command",
    )

    # stats subcommand
    stats_parser = subparsers.add_parser(
        "stats", help="Plot the distribution of a metric"
    )
    _add_stats_args(stats_parser)

    # serve subcommand
    serve_parser = subparsers.add_parser(
        "serve", help="Watch and steer the runs of a directory in a browser"
    )
    serve_parser.add_argument(
        "directory",
        metavar="DIR",
        nargs="?",
        default=None,
        help="The directory to watch: its configuration, and the cwd a run "
        "started from here gets (default: the working directory)",
    )
    serve_parser.add_argument(
        "--workspace",
        default=None,
        metavar="DIR",
        help="Keep Yuclid's state in DIR instead of ./.yuclid: the runs it "
        "records, their logs and their captured output",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=0,
        metavar="N",
        help="Port to listen on, on the loopback interface (default: any free one)",
    )
    serve_parser.add_argument(
        "--open",
        default=False,
        action="store_true",
        help="Open the page in a browser once the server is up",
    )
    serve_parser.add_argument(
        "-b",
        "--background",
        default=False,
        action="store_true",
        help="Start the server detached and return as soon as it is up, "
        "instead of occupying the terminal until it is stopped",
    )
    serve_parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Start even if another server is already watching this directory",
    )
    serve_parser.add_argument(
        "--quiet",
        default=False,
        action="store_true",
        help="Print only warnings and errors, not the address the page is on",
    )

    # finish subcommand
    finish_parser = subparsers.add_parser(
        "finish", help="Run again whatever a previous run did not record"
    )
    finish_parser.add_argument("run", metavar="ID", help="The run to finish")
    finish_parser.add_argument(
        "--workspace",
        default=None,
        metavar="DIR",
        help="Read the runs recorded in DIR instead of ./.yuclid",
    )

    # replay subcommand
    replay_parser = subparsers.add_parser(
        "replay", help="Run the points a previous run measured, again"
    )
    replay_parser.add_argument("run", metavar="ID", help="The run to replay")
    replay_parser.add_argument(
        "--workspace",
        default=None,
        metavar="DIR",
        help="Read the runs recorded in DIR instead of ./.yuclid",
    )
    replay_parser.add_argument(
        "--no-steering",
        default=False,
        action="store_true",
        help="Run the original command in full, rather than the points it "
        "ended up measuring",
    )
    replay_parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=None,
        metavar="N",
        help="Repeat each point N times, instead of as many times as the "
        "original asked for",
    )
    replay_parser.add_argument(
        "-s",
        "--select",
        nargs="*",
        default=None,
        help="Keep only the points matching these dim=values, so a replay can "
        "cover part of what the original did",
    )
    replay_parser.add_argument(
        "--parallel-trials",
        type=int,
        default=None,
        nargs="?",
        const=-1,
        metavar="N",
        help="Run trials in parallel, instead of as the original did",
    )

    # skills subcommand
    skills_parser = subparsers.add_parser(
        "skills", help="Install or uninstall Yuclid's agent skills"
    )
    skills_actions = skills_parser.add_subparsers(
        dest="skills_action", required=True
    )
    for action in ("install", "uninstall"):
        action_parser = skills_actions.add_parser(
            action, help="{} Yuclid's bundled skills".format(action.capitalize())
        )
        destination = action_parser.add_mutually_exclusive_group(required=True)
        destination.add_argument(
            "--agent",
            nargs="+",
            choices=["codex", "claude"],
            help="Install for one or more known agents in their user-wide "
            "skills directory, e.g. --agent claude codex",
        )
        destination.add_argument(
            "--directory",
            "--dir",
            metavar="DIR",
            help="The skills directory itself, including the 'skills' component; "
            "for example ~/.agents/skills, not ~/.agents",
        )
        action_parser.add_argument(
            "--force",
            action="store_true",
            help="Replace or remove skills that have local modifications",
        )

    _add_steering_parsers(subparsers)

    parser.add_argument("--version", action="version", version="yuclid " + __version__)

    return parser


STEERING = ["pause", "resume", "stop", "kill", "drop", "add", "repeat", "order"]


def _add_steering_parsers(subparsers):
    """The commands that change a run while it is under way.

    Each one addresses the run in progress, which is why none of them takes a
    file: `--run` is only needed when more than one is going at once.
    """
    status_parser = subparsers.add_parser(
        "status", help="Show how far the run in progress has got"
    )
    status_parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="Keep the line up to date until the run ends",
    )

    described = {
        "pause": "Stop as soon as the commands in flight are done",
        "resume": "Carry on after a pause",
        "stop": "End the run now, killing whatever is running",
        "kill": "Abandon what is running right now",
        "drop": "Take points out of what is left to run",
        "add": "Put points into what is left to run",
        "repeat": "Change how many repetitions the remaining points ask for",
        "order": "Reorder the points that have not run yet",
    }
    parsers = {"status": status_parser}
    for name in STEERING:
        parsers[name] = subparsers.add_parser(name, help=described[name])

    parsers["kill"].add_argument(
        "scope",
        choices=["point", "rep", "hold"],
        help="'point' abandons the point entirely, 'rep' abandons only the "
        "repetition in flight, 'hold' kills it without abandoning it — it "
        "runs again once a pause on the point is resumed. There is no "
        "default: a kill says what it does to what it reaches",
    )
    parsers["kill"].add_argument(
        "point",
        nargs="*",
        default=[],
        help="Which point to kill, as dim=value pairs. Defaults to whatever is "
        "running, which is all there is unless trials run in parallel",
    )
    for name in ("drop", "add"):
        parsers[name].add_argument(
            "coords",
            metavar="DIM=VALUES",
            nargs="+",
            help="The points to {}, as dim=value1,value2. Naming one dimension "
            "means a whole slice, naming them all means a single point".format(name),
        )
    parsers["repeat"].add_argument("value", type=int, metavar="N")
    parsers["order"].add_argument("dimensions", metavar="DIM", nargs="+")

    for name, p in parsers.items():
        p.add_argument(
        "--workspace",
        default=None,
        metavar="DIR",
        help="Keep Yuclid's state in DIR instead of ./.yuclid: the runs it "
        "records, their logs and their captured output",
    )
        p.add_argument(
            "--run",
            default=None,
            metavar="ID",
            help="Which run to address. Needed only when several are in progress",
        )


def _add_plot_args(p):
    """Register the shared argument set used by both `plot` and `tplot`."""
    p.add_argument(
        "files", metavar="FILES", type=str, nargs="+", help="JSON Lines or CSV files"
    )
    p.add_argument("-x", default=None, help="X-axis column name")
    p.add_argument("-y", nargs="*", default=[], help="Y-axis column names")
    p.add_argument("-z", help="Grouping column name")
    p.add_argument(
        "-X",
        "--x-norm",
        nargs="*",
        help="Normalize each x-group w.r.t. a value per group",
    )
    p.add_argument(
        "-Z",
        "--z-norm",
        nargs="*",
        help="Normalize each z-group w.r.t. a value per group",
    )
    p.add_argument(
        "-R",
        "--ref-norm",
        nargs="*",
        help="Normalize all values w.r.t. a single reference",
    )
    p.add_argument(
        "-r",
        "--norm-reverse",
        action="store_true",
        default=False,
        help="Reverse normalization. E.g. a/x instead of x/a",
    )
    p.add_argument(
        "-m",
        "--spread-measure",
        default="pi,95",
        help="Measure of dispersion. Default: pi,95. Available: none or {}".format(
            " - ".join(yuclid.spread.available)
        ),
    )
    p.add_argument(
        "-l",
        "--lines",
        action="store_true",
        default=False,
        help="Plot with lines instead of bars",
    )
    p.add_argument(
        "-g",
        "--geomean",
        action="store_true",
        default=False,
        help="Include a geomean summary",
    )
    p.add_argument(
        "-f",
        "--filter",
        nargs="*",
        default=None,
        help="Filter dimension with explicit values. E.g. -f a=1 b=value",
    )
    p.add_argument(
        "-u",
        "--unit",
        default=None,
        help="Unit of measurement for the Y-axis",
    )
    p.add_argument(
        "-d",
        "--digits",
        type=int,
        default=2,
        help="Number of digits to display on annotations (default: %(default)s)",
    )
    p.add_argument(
        "--colorblind",
        action="store_true",
        default=False,
        help="Enable colorblind palette",
    )
    p.add_argument(
        "--rescale",
        type=float,
        default=1.0,
        help="Rescale Y-axis values by multiplying by this number",
    )
    p.add_argument(
        "-A",
        "--annotate",
        action="store_true",
        default=False,
        help="Annotate Y values on each bar or point in the plot",
    )
    p.add_argument(
        "--annotate-max",
        action="store_true",
        default=False,
        help="Annotate only the maximum Y value in each group",
    )
    p.add_argument(
        "--annotate-min",
        action="store_true",
        default=False,
        help="Annotate only the minimum Y value in each group",
    )
    p.add_argument(
        "--no-merge-inputs",
        action="store_true",
        default=False,
        help="Treat each input file separately instead of merging them (default: merged). "
        "A new index column 'file' will be created for each input file.",
    )
    p.add_argument(
        "-L",
        "--lock-dims",
        nargs="*",
        default=[],
        help="Lock a free dimension with explicit values. E.g. -L a=1 b=value",
    )
    p.add_argument(
        "-C",
        "--combine",
        nargs="*",
        default=[],
        help="Combine dimensions into a single one via cartesian product. "
        "E.g. -C a,b c,d will create two new dimensions 'a_b' and 'c_d'.",
    )
    p.add_argument(
        "--array-reduce",
        default=None,
        choices=["mean", "median", "min", "max", "sum"],
        help="Reduce array-valued metrics to a scalar using the given function. "
        "If not set, array elements are exploded into separate rows (samples).",
    )


def _add_describe_args(p):
    p.add_argument(
        "files",
        metavar="FILES",
        type=str,
        nargs="*",
        default=[],
        help="JSON Lines or CSV files",
    )
    p.add_argument(
        "-y",
        nargs="*",
        default=[],
        help="Treat these columns as the metrics (auto-detected if omitted)",
    )
    p.add_argument(
        "-f",
        "--filter",
        nargs="*",
        default=None,
        help="Describe only these values. E.g. -f a=1 b=value",
    )
    p.add_argument(
        "-C",
        "--combine",
        nargs="*",
        default=[],
        help="Combine dimensions into a single one via cartesian product",
    )
    p.add_argument(
        "--missing",
        action="store_true",
        default=False,
        help="List every missing combination instead of summarizing them",
    )
    p.add_argument(
        "--points",
        action="store_true",
        default=False,
        help="Write the points these results hold, as a file `yuclid run "
        "--points` can take, instead of describing them",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="Write --points here rather than to standard output",
    )
    p.add_argument(
        "--no-merge-inputs",
        action="store_true",
        default=False,
        help="Treat each input file separately instead of merging them",
    )
    p.set_defaults(array_reduce=None)


def _add_stats_args(p):
    p.add_argument(
        "files", metavar="FILES", type=str, nargs="+", help="JSON Lines or CSV files"
    )
    p.add_argument("-y", nargs="*", default=[], help="Metric column(s) to analyze (auto-detected if omitted)")
    p.add_argument("-z", help="Grouping column — one distribution per value")
    p.add_argument(
        "-m",
        "--distribution",
        default="normal",
        choices=["none", "normal", "lognormal", "exponential", "gamma", "beta"],
        help="Statistical distribution to fit and overlay (default: normal)",
    )
    p.add_argument(
        "-f",
        "--filter",
        nargs="*",
        default=None,
        help="Filter rows by dimension values. E.g. -f a=1 b=value",
    )
    p.add_argument(
        "--no-merge-inputs",
        action="store_true",
        default=False,
        help="Treat each input file separately instead of merging them",
    )
    p.add_argument(
        "-C",
        "--combine",
        nargs="*",
        default=[],
        help="Combine dimensions into a single one via cartesian product",
    )
    p.add_argument(
        "-L",
        "--lock-dims",
        nargs="*",
        default=[],
        help="Lock a free dimension with explicit values. E.g. -L a=1 b=value",
    )
    p.add_argument(
        "--colorblind",
        action="store_true",
        default=False,
        help="Enable colorblind palette",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=None,
        metavar="N",
        help="Number of histogram bins (default: auto)",
    )
    p.add_argument(
        "-u",
        "--unit",
        default=None,
        help="Unit of measurement for the x-axis",
    )
    p.add_argument(
        "--mean",
        action="store_true",
        default=False,
        help="Show a vertical line at the mean of each group",
    )
    p.add_argument(
        "--median",
        action="store_true",
        default=False,
        help="Show a vertical line at the median of each group",
    )


def main():
    parser = get_parser()
    args = parser.parse_args()
    yuclid.log.init()

    try:
        dispatch(args)
    except KeyboardInterrupt:
        # Ctrl-C is how a run is meant to be stopped, not a crash: say so and
        # leave, with the exit status a shell expects from an interrupted
        # program. Whatever was under way has already recorded itself
        yuclid.log.report(yuclid.log.LogLevel.INFO, "interrupted")
        return 130


def dispatch(args):
    if args.command == "run":
        yuclid.run.launch(args)
    elif args.command == "serve":
        from yuclid import server as _server

        _server.launch(args)
    elif args.command == "replay":
        from yuclid import steer as _steer

        _steer.launch_replay(args, get_parser())
    elif args.command == "finish":
        from yuclid import steer as _steer

        _steer.launch_finish(args, get_parser())
    elif args.command in ("runs", "status") or args.command in STEERING:
        from yuclid import steer as _steer

        if args.command == "runs":
            _steer.launch_runs(args)
        elif args.command == "status":
            _steer.launch_status(args)
        else:
            _steer.launch_operation(args)
    elif args.command == "plot":
        yuclid.plot.launch(args)
    elif args.command == "tplot":
        yuclid.tplot.launch(args)
    elif args.command == "describe":
        from yuclid import describe as _describe
        _describe.launch(args)
    elif args.command == "stats":
        from yuclid import stats as _stats
        _stats.launch(args)
    elif args.command == "skills":
        from yuclid import skill_install as _skill_install

        _skill_install.launch(args)


if __name__ == "__main__":
    main()

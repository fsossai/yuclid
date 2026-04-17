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

    parser.add_argument(
        "--ignore-errors",
        default=False,
        action="store_true",
        help="Yuclid will not abort on any errors unless fatal",
    )

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run experiments and collect data")
    run_parser.add_argument(
        "-i",
        "--inputs",
        default=["yuclid.json"],
        nargs="*",
        help="Specify one or more configuration files. Default is 'yuclid.json'. "
        "Objects and lists will be joined",
    )
    run_parser.add_argument(
        "-r",
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
        "--temp-dir",
        default=".yuclid",
        help="Directory where temporary file will be saved",
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
        default=False,
        action="store_true",
        help="Skip the setup phase and run trials directly",
    )

    # plot subcommand — GUI
    plot_parser = subparsers.add_parser("plot", help="Plot data in a GUI")
    _add_plot_args(plot_parser)

    # tplot subcommand — terminal version of plot
    tplot_parser = subparsers.add_parser(
        "tplot", help="Plot data interactively in the terminal"
    )
    _add_plot_args(tplot_parser)

    parser.add_argument("--version", action="version", version="yuclid " + __version__)

    return parser


def _add_plot_args(p):
    """Register the shared argument set used by both `plot` and `tplot`."""
    p.add_argument(
        "files", metavar="FILES", type=str, nargs="+", help="JSON Lines or CSV files"
    )
    p.add_argument("-x", required=True, help="X-axis column name")
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
        default=2,
        help="Number of digits to display on annotations",
    )
    p.add_argument(
        "--colorblind",
        action="store_true",
        default=False,
        help="Enable colorblind palette",
    )
    p.add_argument(
        "--show-missing",
        action="store_true",
        default=False,
        help="Show missing experiments if any",
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


def main():
    parser = get_parser()
    args = parser.parse_args()
    yuclid.log.init(ignore_errors=args.ignore_errors)

    if args.command == "run":
        yuclid.run.launch(args)
    elif args.command == "plot":
        yuclid.plot.launch(args)
    elif args.command == "tplot":
        yuclid.tplot.launch(args)


if __name__ == "__main__":
    main()

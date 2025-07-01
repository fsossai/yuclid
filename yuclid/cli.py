from yuclid import __version__
import yuclid.spread
import yuclid.plot
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
        help="Specify one or more configuration files. Default is 'yuclid.json'",
    )
    run_parser.add_argument(
        "-r",
        "--order",
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
        help="Specify a list of preset names to run",
    )
    run_parser.add_argument(
        "-s",
        "--select",
        nargs="*",
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
        "-m",
        "--metrics",
        nargs="*",
        default=None,
        help="Specify a list of metrics to compute among the registered ones. If empty, all metrics will be computed.",
    )

    # plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Plot data in a GUI")
    plot_parser.add_argument(
        "files", metavar="FILES", type=str, nargs="+", help="JSON Lines or CSV files"
    )
    plot_parser.add_argument("-x", required=True, help="X-axis column name")
    plot_parser.add_argument("-y", nargs="*", help="Y-axis column names")
    plot_parser.add_argument("-z", help="Grouping column name")
    plot_parser.add_argument(
        "-X",
        "--x-norm",
        nargs="*",
        help="Normalize each x-group w.r.t. a value per group",
    )
    plot_parser.add_argument(
        "-Z",
        "--z-norm",
        nargs="*",
        help="Normalize each z-group w.r.t. a value per group",
    )
    plot_parser.add_argument(
        "-R",
        "--ref-norm",
        nargs="*",
        help="Normalize all values w.r.t. a single reference",
    )
    plot_parser.add_argument(
        "-r",
        "--norm-reverse",
        action="store_true",
        default=False,
        help="Reverse normalization. E.g. a/x instead of x/a",
    )
    plot_parser.add_argument(
        "-m",
        "--spread-measure",
        default="pi,95",
        help="Measure of dispersion. Default: pi,95. Available: none or {}".format(
            " - ".join(yuclid.spread.available)
        ),
    )
    plot_parser.add_argument(
        "--rsync-interval",
        metavar="S",
        type=float,
        default=5,
        help="[seconds] Remote synchronization interval",
    )
    plot_parser.add_argument(
        "-l",
        "--lines",
        action="store_true",
        default=False,
        help="Plot with lines instead of bars",
    )
    plot_parser.add_argument(
        "-g",
        "--geomean",
        action="store_true",
        default=False,
        help="Include a geomean summary",
    )
    plot_parser.add_argument(
        "-f",
        "--filter",
        nargs="*",
        default=None,
        help="Filter dimension with explicit values. E.g. -f a=1 b=value",
    )
    plot_parser.add_argument(
        "-u",
        "--unit",
        default=None,
        help="Unit of measurement for the Y-axis",
    )
    plot_parser.add_argument(
        "--colorblind",
        action="store_true",
        default=False,
        help="Enable colorblind palette",
    )
    plot_parser.add_argument(
        "--show-missing",
        action="store_true",
        default=False,
        help="Show missing experiments if any",
    )
    plot_parser.add_argument(
        "--rescale",
        type=float,
        default=1.0,
        help="Rescale Y-axis values by multiplying by this number",
    )
    plot_parser.add_argument(
        "-A",
        "--annotate",
        action="store_true",
        default=False,
        help="Annotate Y values on each bar or point in the plot",
    )
    plot_parser.add_argument(
        "--annotate-max",
        action="store_true",
        default=False,
        help="Annotate only the maximum Y value in each group",
    )
    plot_parser.add_argument(
        "--annotate-min",
        action="store_true",
        default=False,
        help="Annotate only the minimum Y value in each group",
    )
    plot_parser.add_argument(
        "--no-merge-inputs",
        action="store_true",
        default=False,
        help="Treat each input file separately instead of merging them (default: merged). A new index column 'file' will be created for each input file.",
    )

    parser.add_argument("--version", action="version", version="yuclid " + __version__)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    yuclid.log.init(ignore_errors=args.ignore_errors)

    if args.command == "run":
        yuclid.run.launch(args)
    elif args.command == "plot":
        yuclid.plot.launch(args)


if __name__ == "__main__":
    main()

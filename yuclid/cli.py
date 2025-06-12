from yuclid import __version__
import yuclid.spread
import yuclid.plot
import yuclid.run
import yuclid.log
import argparse


def main():
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
        default=[],
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
        help="Specify a list of preset names to run",
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

    # plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Plot data in a GUI")
    plot_parser.add_argument(
        "files", metavar="FILES", type=str, nargs="+", help="JSON Lines or CSV files"
    )
    plot_parser.add_argument("-x", required=True, help="X-axis column name")
    plot_parser.add_argument("-y", nargs="*", help="Y-axis column names")
    plot_parser.add_argument("-z", help="Grouping column name")
    plot_parser.add_argument(
        "-n", "--normalize", default=None, help="Normalize w.r.t. a value in -z"
    )
    plot_parser.add_argument(
        "-s", "--speedup", default=None, help="Reverse-normalize w.r.t. a value in -z"
    )
    plot_parser.add_argument(
        "-m",
        "--spread-measure",
        default="pi95",
        help="Measure of dispersion. Default: pi95. Available: {}".format(
            ", ".join(yuclid.spread.available)
        ),
    )
    plot_parser.add_argument(
        "-r",
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
        "--annotate",
        action="store_true",
        default=False,
        help="Annotate Y values on each bar or point in the plot",
    )

    parser.add_argument("--version", action="version", version="yuclid " + __version__)

    args = parser.parse_args()
    yuclid.log.init(ignore_errors=args.ignore_errors)

    if args.command == "run":
        yuclid.run.launch(args)
    elif args.command == "plot":
        yuclid.plot.launch(args)


if __name__ == "__main__":
    main()

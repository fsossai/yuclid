from datetime import datetime
import sys

_state = {}


class LogLevel:
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4
    HINT = 5


def init():
    _state["style"] = {
        "none": "\033[0m",
        "yellow": "\033[93m",
        "green": "\033[92m",
        "red": "\033[91m",
        "bold": "\033[1;97m",
        "purple": "\033[95m",
    }
    if not sys.stdout.isatty():
        for k, v in _state["style"].items():
            _state["style"][k] = ""

    style = _state["style"]
    _state["level_prefix"] = {
        LogLevel.INFO: "{}INFO{}".format(style["green"], style["none"]),
        LogLevel.WARNING: "{}WARNING{}".format(style["yellow"], style["none"]),
        LogLevel.ERROR: "{}ERROR{}".format(style["red"], style["none"]),
        LogLevel.FATAL: "{}FATAL{}".format(style["red"], style["none"]),
        LogLevel.HINT: "{}HINT{}".format(style.get("purple", ""), style["none"]),
    }


def yprint(level, *args, **kwargs):
    style = _state["style"]
    yuclid_prefix = "{}yuclid{}".format(style["bold"], style["none"])
    timestamp = "{:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    level_prefix = _state["level_prefix"].get(level, "UNKNOWN")
    kwargs["sep"] = kwargs.get("sep", ": ")
    print(yuclid_prefix, timestamp, level_prefix, *args, **kwargs)


def report(level, *args, **kwargs):
    """Say something, and stop the program when what was said is fatal.

    An ERROR ends the run by default, because most of them mean yuclid was
    asked for something it cannot do. The failures that happen while
    experiments are running are different — one bad point is not a reason to
    throw away the others — so those pass `fatal=False` and the run goes on.
    """
    hint = kwargs.pop("hint", None)
    fatal = kwargs.pop("fatal", None)
    yprint(level, *args, **kwargs)
    if hint is not None:
        # a hint may carry several lines: each one is a hint of its own, so
        # that every line printed says what it is
        lines = hint if isinstance(hint, (list, tuple)) else str(hint).splitlines()
        for line in lines:
            yprint(LogLevel.HINT, line)
    if level == LogLevel.FATAL:
        sys.exit(2)
    if level == LogLevel.ERROR and (fatal is None or fatal):
        sys.exit(1)

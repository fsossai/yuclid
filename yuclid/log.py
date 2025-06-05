from datetime import datetime
import sys

class LogLevel:
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4

class TextColors(dict):
    def __init__(self):
        if sys.stdout.isatty():
            self.all = {
                "none": "\033[0m",
                "yellow": "\033[93m",
                "green": "\033[92m",
                "red": "\033[91m",
                "bold": "\033[1;97m",
            }
        else:
            self.all = dict()

    def get_color(self, x):
        return self.all.get(x, "")

    __getattr__ = get_color

_state = {
    "color": None,
    "ignore_errors": None
}

def init(ignore_errors):
    _state["color"] = TextColors()
    _state["ignore_errors"] = ignore_errors

def report(level, *pargs, **kwargs):
    color = _state["color"]
    timestamp = "{:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    log_prefix = {
        LogLevel.INFO: f"{color.green}INFO{color.none}",
        LogLevel.WARNING: f"{color.yellow}WARNING{color.none}",
        LogLevel.ERROR: f"{color.red}ERROR{color.none}",
        LogLevel.FATAL: f"{color.red}FATAL{color.none}",
    }.get(level, "UNKNOWN")
    kwargs["sep"] = kwargs.get("sep", ": ")
    print(f"{color.bold}yuclid{color.none}", timestamp, log_prefix, *pargs, **kwargs)
    if level == LogLevel.FATAL:
        sys.exit(2)
    if not _state["ignore_errors"] and level == LogLevel.ERROR:
        sys.exit(1)

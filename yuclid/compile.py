"""Everything behind `yuclid run --compile`.

The compiled script is a flat sequence of commands: every point of the space,
every repetition and every trial is unrolled while compiling, so the script
holds no loop and no branch, and needs neither yuclid nor the configuration to
run.
"""

from yuclid.log import LogLevel, report
from yuclid import __version__
from datetime import datetime
import yuclid.run
import json
import os


def sh_quote(text):
    """Quote a string so a POSIX shell reads it back verbatim."""
    return "'" + str(text).replace("'", "'\\''") + "'"


def sh_in_quotes(text):
    """Escape a string for use inside a double-quoted shell word."""
    for char in ["\\", '"', "$", "`"]:
        text = text.replace(char, "\\" + char)
    return text


def sh_env_value(text):
    """An env value for a double-quoted `export`, expanded as yuclid expands it.

    A reference is left for the shell to resolve, since the script's exports
    run in the same order the configuration sets them. Everything else yuclid
    treats as text is protected, so quotes survive and a command substitution
    stays the string it was rather than becoming something that runs.
    """
    for char in ["\\", '"', "`"]:
        text = text.replace(char, "\\" + char)
    return text.replace("$(", "\\$(").replace("$$", "\\$")


def sh_printf_literal(text):
    """Escape text that goes into a printf format, where a % is a directive."""
    return text.replace("%", "%%")


def csv_quote(text):
    """Quote a CSV field the way csv.writer would."""
    text = str(text)
    if any(c in text for c in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


BOLD = "\\033[1m"
GREEN = "\\033[92m"
BLUE = "\\033[94m"
YELLOW = "\\033[93m"
PLAIN = "\\033[0m"


class ScriptWriter:
    """Collects the shell script produced by --compile."""

    def __init__(self, path):
        self.path = path
        self.lines = []
        self.total = 0

    def blank(self):
        if len(self.lines) > 0 and self.lines[-1] != "":
            self.lines.append("")

    def comment(self, text=""):
        self.lines.append(("# " + text).rstrip())

    def section(self, title):
        self.blank()
        self.comment("--- {} ".format(title).ljust(72, "-"))

    def command(self, text):
        self.lines.append(text)

    def progress(self, text, colour=GREEN, argument=None):
        """Emit a line that reports where the script has got to.

        It goes to standard error, so redirecting the script's output leaves
        the progress visible and keeps the redirected stream clean. Values
        that come from the configuration are passed as arguments rather than
        interpolated into the format, where a '%' would break printf.
        """
        line = "printf '{}yuclid{} %s {}{}{}: {}\\n' \"$(date +%H:%M:%S)\"".format(
            BOLD, PLAIN, colour, "INFO", PLAIN, text
        )
        if argument is not None:
            line += " " + sh_quote(argument)
        self.command(line + " >&2")

    def save(self):
        directory = os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w") as f:
            f.write("\n".join(self.lines) + "\n")
        os.chmod(self.path, 0o755)


def compile_csv_record(script, coordinates, metric_names, columns):
    """Emit the CSV row of one point, with an empty cell per absent metric."""
    files = ['"$R".m{}'.format(k) for k in range(len(metric_names))]
    slot = {name: k + 1 for k, name in enumerate(metric_names)}
    named = dict(coordinates)

    fields, values = [], []
    for column in columns:
        if column in named:
            fields.append(csv_quote(named[column]))
        elif column in slot:
            fields.append("%s")
            values.append('(${0}=="" ? "nan" : ${0})'.format(slot[column]))
        else:
            # a metric that does not apply at this point
            fields.append("")
    program = 'BEGIN {{ FS="\\t" }} {{ printf "{}\\n"{} }}'.format(
        ",".join(fields).replace('"', '\\"'),
        (", " + ", ".join(values)) if values else "",
    )
    script.command(
        'paste {} | awk {} >> "$OUTPUT"'.format(" ".join(files), sh_quote(program))
    )


def compile_record(script, coordinates, metric_names, folded, fmt, columns):
    """Emit the commands that turn a point's metric files into records.

    Exploded mode zips the per-metric sample files row by row, which is what
    `paste` does, and pads the short ones with NaN. Folded mode joins each
    file into one array.
    """
    if len(metric_names) == 0:
        script.comment("no metric applies here: nothing to record")
        return

    if fmt == "csv":
        compile_csv_record(script, coordinates, metric_names, columns)
        return

    files = ['"$R".m{}'.format(k) for k in range(len(metric_names))]
    prefix = ", ".join(
        "{}: {}".format(json.dumps(str(dim)), json.dumps(str(name)))
        for dim, name in coordinates
    )

    # paste zips the files and pads the short ones with empty fields, which is
    # where the NaNs come from. The awk program is unrolled over the metrics,
    # so it holds no loop either.
    if folded:
        fields = ", ".join(
            "{}: [%s]".format(json.dumps(name)) for name in metric_names
        )
        collect = " ".join(
            's{0} = s{0} sep (${0}=="" ? "NaN" : ${0});'.format(k + 1)
            for k in range(len(metric_names))
        )
        values = ", ".join("s{}".format(k + 1) for k in range(len(metric_names)))
        program = (
            'BEGIN {{ FS="\\t" }} '
            "{{ {} sep = \", \" }} "
            'END {{ printf "{{{}, {}}}\\n", {} }}'
        ).format(
            collect,
            prefix.replace('"', '\\"'),
            fields.replace('"', '\\"'),
            values,
        )
    else:
        fields = ", ".join("{}: %s".format(json.dumps(name)) for name in metric_names)
        values = ", ".join(
            '(${0}=="" ? "NaN" : ${0})'.format(k + 1)
            for k in range(len(metric_names))
        )
        program = 'BEGIN {{ FS="\\t" }} {{ printf "{{{}, {}}}\\n", {} }}'.format(
            prefix.replace('"', '\\"'), fields.replace('"', '\\"'), values
        )

    script.command(
        'paste {} | awk {} >> "$OUTPUT"'.format(" ".join(files), sh_quote(program))
    )


def compile_point_trials(settings, data, execution, i, point, script):
    run = yuclid.run
    point_map = {key: x for key, x in zip(execution["order"], point)}
    compatible_trials, compatible_metrics = run.get_compatible_trials_and_metrics(
        data, point, execution
    )
    coordinates = [(key, x["name"]) for key, x in point_map.items()]

    i_padded = str(i).zfill(len(str(execution["subspace_size"])))
    repeat = settings["repeat"]
    # a compiled script has no plan to steer, so its counter is plain arithmetic
    total, base = execution["subspace_size"] * repeat, (i - 1) * repeat

    for rep in range(repeat):
        rep_suffix = "_rep{}".format(rep) if repeat > 1 else ""
        # the counter is printed before the block runs, so it says what is
        # already finished, exactly as `yuclid run` does
        counter = run.get_progress(base + rep, total)
        script.section("{} {}{}".format(counter, run.point_to_string(point), rep_suffix))
        script.progress(
            "{}{}{} %s".format(BLUE, sh_printf_literal(counter), PLAIN),
            argument=run.point_to_string(point),
        )

        stem = "{}.{}{}".format(i_padded, run.point_to_string(point), rep_suffix)
        script.command('R="$WORK/{}"'.format(sh_in_quotes(stem)))

        metric_slots = dict()
        for j, trial in enumerate(compatible_trials):
            for metric in compatible_metrics:
                if trial["metrics"] is None or metric["name"] in trial["metrics"]:
                    metric_slots[metric["name"]] = j
            command = run.substitute_global_yvars(
                trial["command"], execution["subspace"]
            )
            command = run.substitute_point_yvars(
                command, point_map, '"$P{}"'.format(j)
            )
            script.command('P{0}="${{R}}_trial{0}"'.format(j))
            script.command('{} > "$P{}".out 2> "$P{}".err'.format(command, j, j))

        names = [m["name"] for m in compatible_metrics]
        for k, metric in enumerate(compatible_metrics):
            point_id = '"$P{}"'.format(metric_slots[metric["name"]])
            command = run.substitute_global_yvars(
                metric["command"], execution["subspace"]
            )
            command = run.substitute_point_yvars(command, point_map, point_id)
            # one sample per line, exactly how yuclid splits a metric's output
            script.command(
                "{} | tr -s '[:space:]' '\\n' | sed '/^$/d' > \"$R\".m{}".format(
                    command, k
                )
            )
        compile_record(
            script,
            coordinates,
            names,
            settings["fold"],
            settings["format"],
            run.record_columns(data, settings, execution["order"]),
        )


def compile_subspace_trials(settings, data, execution, script):
    # remembered so the epilogue can close the counter at its total
    script.total = execution["subspace_size"] * settings["repeat"]
    for i, point in enumerate(execution["subspace_points"], start=1):
        compile_point_trials(settings, data, execution, i, point, script)


def compile_preamble(script, settings, data, columns):
    script.command("#!/bin/sh")
    script.comment(
        "Generated by yuclid {} on {:%Y-%m-%d %H:%M:%S}".format(
            __version__, datetime.now()
        )
    )
    script.comment("from {}".format(", ".join(settings["inputs"])))
    script.comment()
    script.comment("Every point of the space is unrolled, in the order yuclid")
    script.comment("would have run them. Neither yuclid nor the configuration is")
    script.comment("needed to run this script.")
    script.comment()
    if settings["format"] == "csv":
        script.comment("Each execution rewrites the CSV from scratch.")
        script.comment()
    script.comment("Progress is reported on standard error.")
    script.comment()
    script.comment("Override the destinations with the environment:")
    script.comment("  YUCLID_OUTPUT  where the records are appended")
    script.comment("  YUCLID_WORK    where each trial's output is captured")
    script.blank()
    if settings["abort_on_error"]:
        script.command("set -eu")
    else:
        # the default, as in a run: a failing command does not end the script
        script.command("set -u")
    script.blank()
    script.command(
        'OUTPUT="${{YUCLID_OUTPUT:-{}}}"'.format(sh_in_quotes(settings["output"]))
    )
    script.command(
        'WORK="${{YUCLID_WORK:-{}}}"'.format(sh_in_quotes(settings["trials_dir"]))
    )
    script.command('mkdir -p "$WORK"')
    if settings["format"] == "csv":
        # there is no loop-free way to add a header only when the file is new,
        # so a compiled CSV run starts the file afresh every time
        script.command(
            "printf '{}\\n' > \"$OUTPUT\"".format(
                ",".join(csv_quote(c) for c in columns)
            )
        )

    if any(len(group) > 0 for group in data["env"]):
        script.section("environment")
        # exported group by group, in the order the configuration set them, so
        # the script resolves references exactly as a run does
        for group in data["env"]:
            for key, value in group.items():
                script.command('export {}="{}"'.format(key, sh_env_value(value)))


def compile_epilogue(script, settings):
    script.section("done")
    # nothing follows the last block to report its completion, so the script
    # says so itself and the counter reaches its total
    counter = yuclid.run.get_progress(script.total, script.total)
    script.progress("{}{}{}".format(BLUE, sh_printf_literal(counter), PLAIN))
    script.progress("written to %s", colour=YELLOW, argument=settings["output"])


def compile_experiments(settings, data, order, env):
    script = ScriptWriter(settings["compile"])
    compile_preamble(
        script, settings, data, yuclid.run.record_columns(data, settings, order)
    )

    if len(settings["presets"]) > 0:
        for preset_name in settings["presets"]:
            report(LogLevel.INFO, "compiling preset", preset_name)
            script.section("preset {}".format(preset_name))
            script.progress("preset %s", colour=YELLOW, argument=preset_name)
            yuclid.run.run_experiments(
                settings, data, order, env, preset_name, script=script
            )
    else:
        yuclid.run.run_experiments(settings, data, order, env, script=script)

    compile_epilogue(script, settings)
    script.save()
    report(
        LogLevel.INFO,
        "compiled",
        script.path,
        hint="run it with `sh {}`, no yuclid needed".format(script.path),
    )

"""A loopback view of the runs a directory holds.

It owns nothing. Every read comes from the run directories, which is why a run
that finished last week looks the same as one going now and why this can be
started, killed and started again while a run continues undisturbed. The one
thing it cannot do by reading is change a run, so control requests are forwarded
to the run's own socket and answered with what the run said it did.
"""

from yuclid.log import LogLevel, report
from yuclid.run import DEFAULT_INPUTS, remove_duplicates
from yuclid import __version__
import yuclid.workspace as workspace
import yuclid.control as control
import http.server
import subprocess
import threading
import getpass
import signal
import secrets
import json
import time
import sys
import os
import re


PAGE = os.path.join(os.path.dirname(__file__), "web", "index.html")
BODY_LIMIT = 1 << 16
# how many failures the page is sent; the rest are in the terminal and the logs
FAILURES = 20
# how a run is started from here: the same interpreter, whether yuclid is
# installed or being run from a checkout
ENTRY_POINT = "import sys; from yuclid.cli import main; sys.exit(main())"


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "yuclid"

    def log_message(self, *args):
        pass

    # -- plumbing ---------------------------------------------------------

    def respond(self, status, payload, kind="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def permitted(self):
        """Loopback is not a boundary on a shared machine, so check the caller.

        The Host check is what stops a page on the open web from reaching this
        through a name that resolves to 127.0.0.1.
        """
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
            self.respond(403, {"error": "unexpected Host header"})
            return False
        if self.path.startswith("/api/"):
            given = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
            if not secrets.compare_digest(given, self.server.token):
                self.respond(401, {"error": "bad or missing token"})
                return False
        return True

    def do_GET(self):
        if not self.permitted():
            return
        path = self.path.split("?")[0]
        query = dict(
            pair.split("=", 1)
            for pair in self.path.partition("?")[2].split("&")
            if "=" in pair
        )
        try:
            if path in ("/", "/index.html"):
                return self.respond(200, self.page(), "text/html; charset=utf-8")
            if path == "/api/runs":
                return self.respond(200, {"runs": self.summaries()})
            match = re.fullmatch(r"/api/runs/([^/]+)", path)
            if match:
                return self.respond(200, self.run(match.group(1)))
            if path == "/api/config":
                return self.respond(200, self.configuration())
            if path == "/api/usage":
                return self.respond(200, self.usage())
            match = re.fullmatch(r"/api/runs/([^/]+)/progress", path)
            if match:
                return self.respond(200, self.progress(match.group(1), query))
        except FileNotFoundError:
            return self.respond(404, {"error": "no such run"})
        self.respond(404, {"error": "no such resource"})

    def body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > BODY_LIMIT:
            self.respond(413, {"error": "body too long"})
            return None
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self.respond(400, {"error": "body is not JSON"})
            return None

    def do_POST(self):
        if not self.permitted():
            return
        path = self.path.split("?")[0]

        if path == "/api/runs":
            request = self.body()
            if request is None:
                return
            return self.respond(200, self.start_run(request))

        match = re.fullmatch(r"/api/runs/([^/]+)/finish", path)
        if match:
            payload = self.body()
            if payload is None:
                return
            try:
                mode = payload.get("mode") or "finish"
                return self.respond(200, self.finish_run(match.group(1), mode))
            except FileNotFoundError:
                return self.respond(404, {"error": "no such run"})

        match = re.fullmatch(r"/api/runs/([^/]+)/control", path)
        if not match:
            return self.respond(404, {"error": "no such resource"})
        operation = self.body()
        if operation is None:
            return
        try:
            return self.respond(200, self.control(match.group(1), operation))
        except FileNotFoundError:
            return self.respond(404, {"error": "no such run"})

    def do_DELETE(self):
        if not self.permitted():
            return
        path = self.path.split("?")[0]
        if path == "/api/runs":
            return self.respond(200, self.forget_all())
        match = re.fullmatch(r"/api/runs/([^/]+)", path)
        if not match:
            return self.respond(404, {"error": "no such resource"})
        return self.respond(200, self.forget(match.group(1)))

    def forget(self, run_id):
        """Remove a run's record. The measurements it took are left alone."""
        try:
            manifest = self.manifest(run_id)
        except FileNotFoundError:
            return {"error": "no such run"}
        if manifest["state"] == workspace.RUNNING:
            return {"error": "run {} is still going".format(run_id)}
        try:
            workspace.delete_run(self.server.root, run_id)
        except (ValueError, FileNotFoundError) as e:
            return {"error": str(e)}
        except OSError as e:
            return {"error": "cannot remove run {}: {}".format(run_id, e)}
        return {"deleted": [run_id]}

    def forget_all(self):
        """Remove every run that has ended, and say which were left.

        A run still going is not removed: it is writing into the directory
        that would be taken from under it.
        """
        deleted, kept = [], []
        for manifest in workspace.list_runs(self.server.root):
            if manifest["state"] == workspace.RUNNING:
                kept.append(manifest["id"])
                continue
            try:
                workspace.delete_run(self.server.root, manifest["id"])
                deleted.append(manifest["id"])
            except (OSError, ValueError, FileNotFoundError):
                kept.append(manifest["id"])
        return {"deleted": deleted, "kept": kept}

    def do_PUT(self):
        if not self.permitted():
            return
        match = re.fullmatch(r"/api/runs/([^/]+)/name", self.path.split("?")[0])
        if not match:
            return self.respond(404, {"error": "no such resource"})
        payload = self.body()
        if payload is None:
            return
        try:
            manifest = self.manifest(match.group(1))
            name = workspace.write_name(manifest["directory"], payload.get("name"))
        except FileNotFoundError:
            return self.respond(404, {"error": "no such run"})
        except ValueError as e:
            return self.respond(400, {"error": str(e)})
        return self.respond(200, {"name": name})

    # -- what it serves ---------------------------------------------------

    def page(self):
        with open(PAGE, "rb") as f:
            html = f.read()
        html = html.replace(b"__YUCLID_TOKEN__", self.server.token.encode())
        return html.replace(b"__YUCLID_VERSION__", __version__.encode())

    def manifest(self, run_id):
        directory = workspace.run_directory(self.server.root, run_id)
        manifest = workspace.read_manifest(directory)
        if manifest is None:
            raise FileNotFoundError(run_id)
        manifest["state"] = workspace.state_of(manifest)
        manifest["directory"] = directory
        manifest["name"] = workspace.read_name(directory)
        return manifest

    def summaries(self):
        summaries = []
        for manifest in workspace.list_runs(self.server.root):
            seen = scan(manifest["directory"])
            live = manifest["state"] == workspace.RUNNING
            summaries.append(
                {
                    "id": manifest["id"],
                    "name": manifest.get("name"),
                    "state": manifest["state"],
                    "created": manifest.get("created"),
                    "output": manifest.get("output"),
                    "completed": seen["completed"],
                    "total": seen["total"],
                    "live": live,
                    "paused": bool(seen["plan"].get("paused")) if seen["plan"] and live
                    else False,
                    "in_flight": seen["in_flight"] if live else 0,
                    "mood": mood(manifest["state"], seen["plan"], live, seen["failed"]),
                }
            )
        return summaries

    def run(self, run_id):
        manifest = self.manifest(run_id)
        seen = scan(manifest["directory"])
        plan = seen["plan"]
        completed, total = seen["completed"], seen["total"]
        current, failed, in_flight = seen["current"], seen["failed"], seen["in_flight"]

        live = manifest["state"] == workspace.RUNNING
        return {
            "id": manifest["id"],
            "name": manifest.get("name"),
            "state": manifest["state"],
            "created": manifest.get("created"),
            "ended": manifest.get("ended"),
            "output": manifest.get("output"),
            "argv": manifest.get("argv"),
            "replay_of": manifest.get("replay_of"),
            "order": plan["order"] if plan else [],
            "undefined": plan.get("undefined", []) if plan else [],
            "dimensions": dimensions_of(plan),
            "completed": completed,
            "total": total,
            "current": current,
            "command": seen["command"] if live else None,
            "setup_failures": seen["setup_failures"],
            "failures": seen["failures"],
            "failure_count": seen["failure_count"],
            "live": live,
            "in_flight": in_flight if live else 0,
            "paused": bool(plan.get("paused")) if plan and live else False,
            "failed": failed,
            "mood": mood(manifest["state"], plan, live, failed),
            "gaps": gaps_of(plan, live),
        }

    def progress(self, run_id, query):
        manifest = self.manifest(run_id)
        since = int(query.get("since") or 0)
        records = workspace.read_progress(manifest["directory"], since=since)
        return {"records": records, "seq": records[-1]["seq"] if records else since}

    def usage(self):
        """How much room the run directories take, and how many are going.

        Walked on request rather than reported alongside the runs: it means
        touching every trial capture, which is not something to do once a
        second for a figure nobody asked for.
        """
        total, runs, live = 0, 0, 0
        for manifest in workspace.list_runs(self.server.root):
            runs += 1
            if manifest["state"] == workspace.RUNNING:
                live += 1
            for where, _, files in os.walk(manifest["directory"]):
                for name in files:
                    try:
                        # a hard link counts once, and the results it points at
                        # are not this directory's to give back
                        stat = os.lstat(os.path.join(where, name))
                        total += 0 if stat.st_nlink > 1 else stat.st_size
                    except OSError:
                        pass
        return {"runs": runs, "live": live, "bytes": total}

    def configuration(self):
        """The space a new run could be built from, as the config declares it.

        Read with yuclid's own normalizer so that a value written as an object,
        or a whole dimension computed by a `:py` expression, is understood the
        same way a run would understand it. That normalizer reports a bad
        configuration by exiting, which must not happen inside a request, so
        the whole read is fenced off and turned into an answer.
        """
        from yuclid import run as runner

        directory = os.path.dirname(self.server.root)
        inputs = [
            name
            for name in runner.DEFAULT_INPUTS
            if os.path.isfile(os.path.join(directory, name))
        ]
        if len(inputs) == 0:
            return {"error": "no configuration in {}".format(directory)}

        try:
            raw = runner.load_config(os.path.join(directory, inputs[0]))
            space = runner.normalize_space_values(raw.get("space", {}))
            declared = {
                dim: None if values is None else remove_duplicates(
                    [str(v["name"]) for v in values]
                )
                for dim, values in space.items()
            }
            presets = {
                name: resolve_preset(declared, preset)
                for name, preset in sorted((raw.get("presets") or {}).items())
            }
        except SystemExit:
            return {"error": "{} cannot be read; see the terminal".format(inputs[0])}
        except Exception as e:
            return {"error": "{}: {}".format(inputs[0], e)}

        return {"input": inputs[0], "presets": presets, "dimensions": declared}

    def start_run(self, request):
        """Start a run of a subspace the caller chose.

        Every argument is rebuilt here from names checked against the
        configuration, so what arrives is a choice among things that exist
        rather than a command line.
        """
        config = self.configuration()
        if "error" in config:
            return config

        argv = ["run"]
        declared = config["dimensions"]

        select = request.get("select") or {}
        if not isinstance(select, dict):
            return {"error": "select must be an object of dimension to values"}
        selectors = []
        for dim, values in select.items():
            if dim not in declared:
                return {"error": "unknown dimension '{}'".format(dim)}
            known = declared[dim]
            if known is not None:
                unknown = [v for v in values if v not in known]
                if unknown:
                    return {
                        "error": "{} has no value {}".format(dim, ", ".join(unknown))
                    }
            if len(values) == 0:
                return {"error": "no value chosen for {}".format(dim)}
            if any("," in v for v in values):
                return {"error": "a value of {} contains a comma".format(dim)}
            selectors.append("{}={}".format(dim, ",".join(values)))
        if selectors:
            # one -s carrying every selector: the option takes a list, so a
            # second -s would replace the first rather than add to it
            argv += ["-s"] + selectors

        undefined = [d for d, v in declared.items() if v is None and d not in select]
        if undefined:
            return {
                "error": "these dimensions have no values of their own, so a run "
                "has to be given some: {}".format(", ".join(undefined))
            }

        presets = request.get("presets") or []
        unknown = [p for p in presets if p not in config["presets"]]
        if unknown:
            return {"error": "no such preset: {}".format(", ".join(unknown))}
        if presets:
            argv += ["-p"] + list(presets)

        repeat = request.get("repeat")
        repeat = 1 if repeat is None else repeat
        if not isinstance(repeat, int) or not 1 <= repeat <= 10000:
            return {"error": "repeat must be a whole number of repetitions"}
        if repeat > 1:
            argv += ["-r", str(repeat)]

        parallel = request.get("parallel")
        parallel = 1 if parallel is None else parallel
        if not isinstance(parallel, int) or not 1 <= parallel <= 1024:
            return {"error": "parallel must be a whole number of trials at once"}
        if parallel > 1:
            argv += ["--parallel-trials", str(parallel)]

        name = request.get("name") or ""
        try:
            name = workspace.check_name(name)
        except ValueError as e:
            return {"error": str(e)}
        if name:
            argv += ["--name", name]

        answer, _ = self.spawn(argv, os.path.join(self.server.root, "run.log"))
        return answer

    def spawn(self, argv, log_path):
        """Start yuclid, and answer with the name of the run it made.

        A run names itself, so the only way to say which one was started is to
        watch for it to appear — a second or two, mostly spent importing.
        Returns the answer and the child, which the caller may want to keep.
        """
        with self.server.lock:
            before = {m["id"] for m in workspace.list_runs(self.server.root)}
            log = open(log_path, "w")
            child = subprocess.Popen(
                [sys.executable, "-c", ENTRY_POINT] + argv,
                cwd=os.path.dirname(self.server.root),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            fresh = [
                m
                for m in workspace.list_runs(self.server.root)
                if m["id"] not in before
            ]
            if fresh:
                return {"started": fresh[0]["id"]}, child
            if child.poll() is not None:
                break
            time.sleep(0.1)
        return {"error": "the run did not start", "log": log_path}, child

    def finish_run(self, run_id, mode):
        """Start a run from an old one: completing it, or doing it again.

        `finish` resumes into the same results file, so only the points that
        went unmeasured are run. `replay` does the run again as a run of its
        own, with the steering it was given, and `restart` does it again
        without — the difference matters for a run half of whose space was
        dropped while it went. The command is built here from the run's id and
        a mode this method knows, so nothing in a request becomes an argument.
        The new run is detached: `serve` owns no run, and killing it must not
        take one down.
        """
        if mode not in ("finish", "replay", "restart"):
            return {"error": "unknown mode '{}'".format(mode)}

        manifest = self.manifest(run_id)
        if manifest["state"] == workspace.RUNNING:
            return {"error": "run {} is still going".format(run_id)}

        argv = {
            "finish": ["finish", run_id],
            "replay": ["replay", run_id],
            "restart": ["replay", run_id, "--no-steering"],
        }[mode]

        with self.server.lock:
            started = self.server.finishing.get(run_id)
            if started is not None and started.poll() is None:
                return {"error": "run {} was already started again".format(run_id)}

            output = manifest.get("output")
            for other in workspace.live_runs(self.server.root):
                if other.get("output") == output:
                    return {
                        "error": "run {} is already writing that file".format(
                            other["id"]
                        )
                    }

        answer, child = self.spawn(
            argv, os.path.join(manifest["directory"], mode + ".log")
        )
        self.server.finishing[run_id] = child
        answer["mode"] = mode
        return answer

    def control(self, run_id, operation):
        manifest = self.manifest(run_id)
        if manifest["state"] != workspace.RUNNING:
            return {"error": "run {} is {}".format(run_id, manifest["state"])}
        path = os.path.join(manifest["directory"], workspace.CONTROL)
        try:
            return {"effect": control.request(path, operation)}
        except control.ControlError as e:
            return {"error": str(e)}
        except OSError as e:
            return {"error": "cannot reach the run: {}".format(e)}


def ssh_target():
    """`user@host` as it would be typed from somewhere else.

    `hostname -f` rather than the short name, because the machine has to be
    reachable from the laptop doing the forwarding. It resolves names, so it
    can be slow or absent: it gets a moment, and then the short name will do.
    """
    host = ""
    try:
        finished = subprocess.run(
            ["hostname", "-f"], capture_output=True, text=True, timeout=2
        )
        host = finished.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    host = host or workspace.hostname()

    try:
        return "{}@{}".format(getpass.getuser(), host)
    except Exception:
        # no account to name: the host alone is still worth having
        return host


def scan(directory):
    """Everything a run's progress file says, in one pass over it.

    Both the list and the run being displayed need the same answers, and the
    file has to be read either way, so it is read once and read for all of it.
    """
    plan = None
    completed, total, current, failed = 0, 0, None, False
    command, broken, wrong = None, [], []
    # a plan snapshot is only written when the run is steered, so between two
    # operations it says what was true then. What has happened since is in the
    # records after it, and is folded back in before anything reads the plan
    since = dict()
    # a point is in flight between the start of a repetition and its end;
    # repetitions of one point are sequential, so counting them per point says
    # which points are still going without the run having to announce it
    running = dict()

    for record in workspace.read_progress(directory):
        kind = record["type"]
        if kind == "plan":
            plan = record
            # this snapshot is the truth as of now; what came before it is
            # already in it
            since = dict()
        if record.get("total") is not None:
            total = record["total"]
        if kind in ("point.finished", "point.skipped"):
            completed += record.get("repetitions", 1)
            failed = failed or bool(record.get("failed"))
        if kind == "point.started":
            current = record["key"]
            command = None
        if kind == "trial.started":
            command = record.get("command")
        if kind == "setup.failed":
            broken.append(
                {"label": record.get("label"), "command": record.get("command"),
                 "code": record.get("code"), "log": record.get("log"),
                 "said": record.get("said")}
            )
        if kind in ("trial.failed", "metric.failed"):
            wrong.append(
                {
                    "kind": "metric" if kind == "metric.failed" else "trial",
                    "key": record.get("key") or [],
                    "rep": record.get("rep"),
                    "what": record.get("metric") if kind == "metric.failed"
                    else record.get("trial"),
                    "command": record.get("command"),
                    "code": record.get("code"),
                    "log": record.get("log"),
                    "said": record.get("said"),
                    "time": record.get("time"),
                }
            )

        key = tuple(record.get("key") or ())
        if kind == "point.started":
            running[key] = running.get(key, 0) + 1
            since.setdefault(key, {})["started"] = True
        elif kind == "point.finished":
            running[key] = max(0, running.get(key, 0) - 1)
            entry = since.setdefault(key, {})
            entry["done"] = entry.get("done", 0) + record.get("repetitions", 1)
            entry["failed"] = entry.get("failed") or bool(record.get("failed"))
        elif kind == "point.killed":
            running[key] = 0
            entry = since.setdefault(key, {})
            # abandoning one repetition leaves the point to carry on with the
            # rest; abandoning the point is what ends it
            entry["killed"] = entry.get("killed") or record.get("scope") == "point"
        elif kind == "point.skipped":
            # a skipped point was already complete when the plan was written,
            # so there are no repetitions to add — only the fact
            since.setdefault(key, {})["skipped"] = True

    if plan is not None:
        plan = freshen(plan, since)
        failed = failed or any(p["status"] == "failed" for p in plan["points"])

    return {
        "plan": plan,
        "completed": completed,
        "total": total,
        "current": current,
        "failed": failed,
        "command": command,
        "setup_failures": broken,
        # a run where nothing works fails once per point, and every one of them
        # carries what it printed: the most recent are the ones worth sending,
        # and the count says how many there were
        "failures": wrong[-FAILURES:],
        "failure_count": len(wrong),
        "in_flight": sum(1 for count in running.values() if count > 0),
    }


def freshen(plan, since):
    """The plan as it stands now, not as it stood when it was last written.

    Everything derived from a plan — which values are still to be measured,
    which points are gaps, what dropping one would cost — was reading a
    snapshot that is only rewritten when the run is steered. For a run nobody
    steers that is the snapshot taken before the first point ran, so the counts
    the page offered stayed at their starting figures for the whole run.

    A dropped point stays dropped: the plan is the authority on what the run
    intends, and the records only say what became of what it did.
    """
    if not since:
        return plan
    points = []
    for point in plan["points"]:
        seen = since.get(tuple(point["key"]))
        if seen is None or point["status"] == "dropped":
            points.append(point)
            continue
        point = dict(point)
        point["done"] = min(point["target"], point["done"] + seen.get("done", 0))
        if seen.get("killed"):
            point["status"] = "killed"
        elif seen.get("failed"):
            point["status"] = "failed"
        elif seen.get("skipped") or point["done"] >= point["target"]:
            point["status"] = "done"
        elif seen.get("started"):
            point["status"] = "running"
        points.append(point)
    plan = dict(plan)
    plan["points"] = points
    return plan


def mood(state, plan, live, failed):
    """How the run is doing, in one word, worst news first.

    A pause is not news — you asked for it — so a failed point outranks it. A
    run that was stopped ended the way it was told to and is not bad news
    either.
    """
    if state in (workspace.INTERRUPTED, workspace.FAILED):
        return "dead"
    if failed:
        return "failed"
    if live and plan is not None and plan.get("paused"):
        return "paused"
    return "fine"


def resolve_preset(declared, preset):
    """The values a preset names, per dimension, as a run would read them.

    A preset entry is a name, a `*` pattern over the names a dimension has, or
    — where the space leaves a dimension open — a value the preset supplies.
    Resolving it here means the page can say what a preset comes to without
    re-implementing the rule. A dimension the preset does not name is not in
    the result: the preset leaves that one whole.
    """
    resolved = {}
    for dim, items in (preset or {}).items():
        if dim not in declared:
            continue
        known = declared[dim]
        values = []
        for item in items if isinstance(items, list) else [items]:
            text = str(item)
            if known is None:
                values.append(text)
            elif "*" in text:
                pattern = re.compile("^" + re.escape(text).replace("\\*", ".*") + "$")
                values += [name for name in known if pattern.match(name)]
            elif text in known:
                values.append(text)
        resolved[dim] = remove_duplicates(values)
    return resolved


def gaps_of(plan, live):
    """The points of the space this run will not have measured.

    A point still pending is not a gap while the run is going — it is simply
    its turn next — but once the run has ended, one that never came up is as
    absent from the results as one that failed.
    """
    if plan is None:
        return []
    left_behind = {"failed", "killed", "dropped"}
    if not live:
        left_behind |= {"pending", "running"}
    return [
        {"key": p["key"], "status": p["status"]}
        for p in plan["points"]
        if p["status"] in left_behind
    ]


def dimensions_of(plan):
    """Each dimension's values, in the order the space puts them.

    A value counts as dropped once every point carrying it has been: that is
    what makes it something the page can offer to put back. The pending count
    is what dropping it would actually cost, which is worth knowing before
    being asked to confirm it.
    """
    if plan is None:
        return {}
    seen = {dim: [] for dim in plan["order"]}
    dropped = {dim: {} for dim in plan["order"]}
    pending = {dim: {} for dim in plan["order"]}
    for point in plan["points"]:
        for dim, name in zip(plan["order"], point["key"]):
            if name not in seen[dim]:
                seen[dim].append(name)
                dropped[dim][name] = True
                pending[dim][name] = 0
            if point["status"] != "dropped":
                dropped[dim][name] = False
            if point["status"] in ("pending", "running"):
                pending[dim][name] += 1
    return {
        dim: [
            {"value": name, "dropped": dropped[dim][name], "pending": pending[dim][name]}
            for name in names
        ]
        for dim, names in seen.items()
    }


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root, port):
        super().__init__(("127.0.0.1", port), Handler)
        self.root = root
        self.token = secrets.token_urlsafe(24)
        # runs started from here, so a second request cannot start them again
        self.finishing = dict()
        self.lock = threading.Lock()


def launch(args):
    # Serving an empty directory is useful: it can launch the first run from
    # the browser. Use the same workspace creation path as `yuclid run` rather
    # than requiring a run to have happened here already.
    root = workspace.open_root(args.directory)
    directory = os.path.dirname(root)
    if not any(os.path.isfile(os.path.join(directory, name)) for name in DEFAULT_INPUTS):
        report(
            LogLevel.WARNING,
            "no Yuclid configuration found in",
            directory,
            hint="add yuclid.json, yuclid.yaml, or yuclid.yml",
        )
    if not os.path.exists(PAGE):
        report(LogLevel.FATAL, "the web page is missing from the installation", PAGE)

    existing = workspace.read_server(root)
    if existing is not None and not args.force:
        # nothing stops a second one working — they both only read files and
        # forward control — but two tabs of the same runs is rarely the intent
        report(
            LogLevel.FATAL,
            "a server is already watching this directory",
            "http://127.0.0.1:{}/ (pid {})".format(existing["port"], existing["pid"]),
            hint="open that one, or `yuclid serve --force` to start another anyway",
        )

    server = Server(root, args.port)
    if workspace.read_server(root) is None:
        # a forced second server does not take the note over: it stays with
        # the one that was here first, which is the one worth pointing at
        workspace.write_server(root, server.server_port)
    # so that `kill` leaves as tidily as Ctrl-C does
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    # the page carries the token itself, so the URL does not have to: a secret
    # in one ends up in scrollback, shell history and any Referer header
    url = "http://127.0.0.1:{}/".format(server.server_port)
    report(LogLevel.INFO, "watching", root)
    hints = ["stop with Ctrl-C"]
    # bound to loopback, so a machine reached over ssh needs the port brought
    # to where the browser is. Only worth saying when that is the situation:
    # sitting at the machine, the URL above is already the whole story
    if os.environ.get("SSH_CONNECTION"):
        hints.append(
            "for port forwarding: ssh -N -L {0}:127.0.0.1:{0} {1}".format(
                server.server_port, ssh_target()
            )
        )
    report(LogLevel.INFO, "open", url, hint=hints)
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        report(LogLevel.INFO, "stopped")
    finally:
        workspace.clear_server(root)
        server.server_close()

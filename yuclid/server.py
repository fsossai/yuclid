"""A loopback view of the runs a directory holds.

It owns nothing. Every read comes from the run directories, which is why a run
that finished last week looks the same as one going now and why this can be
started, killed and started again while a run continues undisturbed. The one
thing it cannot do by reading is change a run, so control requests are forwarded
to the run's own socket and answered with what the run said it did.
"""

from yuclid.log import LogLevel, report
from yuclid.run import DEFAULT_INPUTS, intended_points, remove_duplicates
from yuclid import __version__
import yuclid.workspace as workspace
import yuclid.control as control
from datetime import datetime
import http.server
import urllib.parse
import subprocess
import threading
import getpass
import shutil
import signal
import secrets
import errno
import json
import time
import sys
import os
import re


PAGE = os.path.join(os.path.dirname(__file__), "web", "index.html")
BODY_LIMIT = 1 << 16
# How many failures the page is sent. A handful is what a reader takes in, and
# a run where everything fails says the same thing three times as clearly as it
# does twenty; the rest are in the terminal and the captures.
FAILURES = 3
# a point list arrives in one request, and a request is not the place to hand
# over an unbounded one
POINT_LIMIT = 20000
# how much of a growing capture file one poll reads: enough that a live tail
# keeps up with a chatty command, not so much that one answer is unbounded
TAIL_CHUNK = 1 << 18
TAIL_STREAMS = ("out", "err", "all")
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
        # unquoted: a value can carry anything a path can, `/` included, which
        # is exactly what a capture's stem is
        query = dict(
            (urllib.parse.unquote(k), urllib.parse.unquote(v))
            for k, v in (
                pair.split("=", 1)
                for pair in self.path.partition("?")[2].split("&")
                if "=" in pair
            )
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
            match = re.fullmatch(r"/api/runs/([^/]+)/tail", path)
            if match:
                return self.respond(200, self.tail(match.group(1), query))
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

        match = re.fullmatch(r"/api/runs/([^/]+)/export", path)
        if match:
            if self.body() is None:
                return
            try:
                return self.respond(200, self.export_run(match.group(1)))
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
        # a literal path of its own: /api/runs/temporary would be read as a run
        # by the route below, and one could be named that
        if path == "/api/temporary":
            return self.respond(200, self.clear_temporary())
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
        html = html.replace(b"__YUCLID_VERSION__", __version__.encode())
        # The workspace, which is where everything shown on the page comes
        # from. With --workspace it is not the working directory, and then it
        # is the one worth naming: two servers on one directory look identical
        # otherwise.
        # the workspace, named by the directory it is about: the `.yuclid` on
        # the end of the usual one says nothing, since every workspace is one
        html = html.replace(b"__YUCLID_ROOT__", self.server.base.encode())
        return html.replace(b"__YUCLID_WHO__", whoami().encode())

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
            "directory": manifest["directory"],
            "mood": mood(manifest["state"], plan, live, failed),
            "gaps": gaps_of(plan, live),
            # what a replay of this run would cover, and how it would be asked
            # for: the page offers these for editing before starting anything
            "replayable": [
                list(key) for key in intended_points(manifest["directory"])
            ],
            "repeat": max(
                [p["target"] for p in plan["points"]] or [1]
            ) if plan else 1,
        }

    def progress(self, run_id, query):
        manifest = self.manifest(run_id)
        since = int(query.get("since") or 0)
        records = workspace.read_progress(manifest["directory"], since=since)
        return {"records": records, "seq": records[-1]["seq"] if records else since}

    def tail(self, run_id, query):
        """A byte range of a trial's captures, so a click can see one that
        already finished.

        `stream`, `since` and `stem` all come from the request — unlike
        everywhere else in this handler, which derives the path itself and
        trusts nothing the caller supplies. Naming a stem is only safe
        because it is checked against every stem this run has ever recorded
        starting a trial for: not a path taken on faith, a name recognised.
        """
        manifest = self.manifest(run_id)
        stream = query.get("stream")
        if stream not in TAIL_STREAMS:
            return {"error": "stream must be one of {}".format(", ".join(TAIL_STREAMS))}
        try:
            since = max(0, int(query.get("since") or 0))
        except ValueError:
            since = 0

        seen = scan(manifest["directory"])
        stem = query.get("stem")
        if stem is None or stem not in seen["stems"]:
            return {"error": "no such capture"}

        try:
            with open("{}.{}".format(stem, stream), "rb") as f:
                size = os.fstat(f.fileno()).st_size
                # a stale offset from a stream that has since been replaced —
                # the same file at a smaller size — starts over rather than
                # erroring
                start = since if since <= size else 0
                f.seek(start)
                raw = f.read(TAIL_CHUNK)
        except FileNotFoundError:
            # the trial has started but this particular file has not been
            # opened yet — nothing to read, not a problem
            return {"text": "", "offset": since, "stem": stem}

        return {
            "text": raw.decode("utf-8", errors="replace"),
            "offset": start + len(raw),
            "stem": stem,
        }

    def export_run(self, run_id):
        """Copy this run's JSONL into the working directory.

        `--workspace` can keep a run's data far from where a person is sitting;
        this puts a copy of it exactly where they are, under its own name, so
        it can be picked up without knowing where the workspace is.
        """
        manifest = self.manifest(run_id)
        output = manifest.get("output")
        if not output or not os.path.exists(output):
            return {"error": "the results of run {} are gone".format(run_id)}
        if not output.lower().endswith(".jsonl"):
            return {"error": "run {} was not recorded as JSONL".format(run_id)}
        destination = os.path.join(self.server.base, os.path.basename(output))
        if os.path.abspath(destination) == os.path.abspath(output):
            return {"path": destination}
        try:
            shutil.copyfile(output, destination)
        except OSError as e:
            return {"error": "cannot write {}: {}".format(destination, e)}
        return {"path": destination}

    def usage(self):
        """How much room the run directories take, and how many are going.

        Walked on request rather than reported alongside the runs: it means
        touching every trial capture, which is not something to do once a
        second for a figure nobody asked for.
        """
        total, scratch, runs, live = 0, 0, 0, 0
        for manifest in workspace.list_runs(self.server.root):
            runs += 1
            if manifest["state"] == workspace.RUNNING:
                live += 1
            else:
                # what clearing the temporary files would give back, which is
                # only ever offered for a run that has ended
                scratch += workspace.temporary_size(manifest["directory"])
            # a hard link counts once, and the results it points at are not
            # this directory's to give back
            total += workspace.directory_size(manifest["directory"])
        return {"runs": runs, "live": live, "bytes": total, "temporary": scratch}

    def clear_temporary(self):
        """Take back the room the captures cost, and keep the runs.

        A run still going is writing into the very directory this would empty,
        so it is left alone and named in the answer.
        """
        cleared, kept, freed = [], [], 0
        for manifest in workspace.list_runs(self.server.root):
            if manifest["state"] == workspace.RUNNING:
                kept.append(manifest["id"])
                continue
            try:
                freed += workspace.clear_temporary(
                    self.server.root, manifest["id"]
                )
                cleared.append(manifest["id"])
            except (OSError, ValueError, FileNotFoundError):
                kept.append(manifest["id"])
        return {"cleared": cleared, "kept": kept, "bytes": freed}

    def configuration(self):
        """The space a new run could be built from, as the config declares it.

        Read with yuclid's own normalizer so that a value written as an object,
        or a whole dimension computed by a `:py` expression, is understood the
        same way a run would understand it. That normalizer reports a bad
        configuration by exiting, which must not happen inside a request, so
        the whole read is fenced off and turned into an answer.
        """
        from yuclid import run as runner

        directory = self.server.base
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
            # a name may be declared several times, once per region of the
            # space, and it is one column and one thing to choose
            metrics = remove_duplicates(
                [m["name"] for m in runner.normalize_metrics(raw.get("metrics") or [])]
            )
        except SystemExit:
            return {"error": "{} cannot be read; see the terminal".format(inputs[0])}
        except Exception as e:
            return {"error": "{}: {}".format(inputs[0], e)}

        return {
            "input": inputs[0],
            "presets": presets,
            "dimensions": declared,
            "metrics": metrics,
        }

    def start_run(self, request):
        """Start a run of a subspace the caller chose.

        Every argument is rebuilt here from names checked against the
        configuration, so what arrives is a choice among things that exist
        rather than a command line.
        """
        config = self.configuration()
        if "error" in config:
            return config

        argv = ["run"] + self.elsewhere()
        declared = config["dimensions"]

        # a point list says exactly what to run, so it stands in for the
        # selection rather than joining it
        points = request.get("points")
        if points is not None:
            checked = check_points(points, declared)
            if isinstance(checked, dict):
                return checked
            if request.get("select") or request.get("presets"):
                return {
                    "error": "a run covers a chosen subspace or a list of "
                    "points, not both"
                }
            path = workspace.write_point_set(
                self.server.root,
                "{:%Y%m%d-%H%M%S}".format(datetime.now()),
                checked,
            )
            argv += ["--points", path]

        select = {} if points is not None else (request.get("select") or {})
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

        undefined = [
            d
            for d, v in declared.items()
            if v is None and d not in select and points is None
        ]
        if undefined:
            return {
                "error": "these dimensions have no values of their own, so a run "
                "has to be given some: {}".format(", ".join(undefined))
            }

        presets = [] if points is not None else (request.get("presets") or [])
        unknown = [p for p in presets if p not in config["presets"]]
        if unknown:
            return {"error": "no such preset: {}".format(", ".join(unknown))}
        if presets:
            argv += ["-p"] + list(presets)

        # the run's own order, not a selection: which dimension the space
        # panel nests outermost afterwards is decided here, once, rather than
        # by whatever order the configuration happened to declare them in
        order = request.get("order")
        if order is not None:
            if not isinstance(order, list) or not all(
                isinstance(d, str) for d in order
            ):
                return {"error": "order must be a list of dimension names"}
            unknown = [d for d in order if d not in declared]
            if unknown:
                return {"error": "unknown dimension in order: {}".format(", ".join(unknown))}
            if len(set(order)) != len(order):
                return {"error": "order names the same dimension more than once"}
            argv += ["--order"] + order

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

        # Choosing metrics is not only about which columns come back: a trial
        # runs when a metric it declares is wanted, so narrowing the metrics
        # can narrow the work as well.
        metrics = request.get("metrics")
        if metrics is not None:
            if not isinstance(metrics, list) or len(metrics) == 0:
                return {"error": "choose at least one metric"}
            unknown = [m for m in metrics if m not in config["metrics"]]
            if unknown:
                return {"error": "no such metric: {}".format(", ".join(unknown))}
            if len(metrics) < len(config["metrics"]):
                argv += ["-m"] + list(metrics)

        name = request.get("name") or ""
        try:
            name = workspace.check_name(name)
        except ValueError as e:
            return {"error": str(e)}
        if name:
            argv += ["--name", name]

        answer, _ = self.spawn(argv, os.path.join(self.server.root, "run.log"))
        return answer

    def elsewhere(self):
        """`--workspace` for a child, when this server keeps its state away.

        Said only when it has to be: a run whose recorded command names an
        absolute workspace cannot be replayed from a copy of the directory
        somewhere else, and the ordinary case has nothing to name.
        """
        beside = os.path.join(self.server.base, workspace.DIRNAME)
        if os.path.abspath(self.server.root) == beside:
            return []
        return ["--workspace", self.server.root]

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
                cwd=self.server.base,
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

    def spawn_continuing(self, run_id, argv, log_path):
        """As `spawn`, for a run continuing in its own directory.

        `finish` writes into the run it is finishing rather than making a new
        one, so there is no fresh id to watch for the way `spawn` waits for
        one. Watching for `state` to turn running is not enough either: a
        `finish` with nothing left to measure can run and end between two
        polls, and the state would never be seen mid-flight. The pid does not
        have that problem — `reopen_run` stamps it with this child's own, and
        once written it stays, whether or not the run has already finished by
        the time this notices.
        """
        directory = workspace.run_directory(self.server.root, run_id)
        with self.server.lock:
            log = open(log_path, "w")
            child = subprocess.Popen(
                [sys.executable, "-c", ENTRY_POINT] + argv,
                cwd=self.server.base,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            manifest = workspace.read_manifest(directory)
            if manifest is not None and manifest.get("pid") == child.pid:
                return {"started": run_id}, child
            if child.poll() is not None:
                break
            time.sleep(0.1)
        return {"error": "the run did not start", "log": log_path}, child

    def finish_run(self, run_id, mode):
        """Start a run from an old one: completing it, or doing it again.

        `finish` continues this same run, writing the points that went
        unmeasured into the same directory and the same results file — it is
        not a run of its own, and nothing new appears in the run list for it.
        `replay` does the run again as a run of its own, with the steering it
        was given, and `restart` does it again without — the difference
        matters for a run half of whose space was dropped while it went. The
        command is built here from the run's id and a mode this method knows,
        so nothing in a request becomes an argument. `replay` and `restart`
        are detached: `serve` owns no run, and killing it must not take one
        down.
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
        }[mode] + self.elsewhere()

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

        log_path = os.path.join(manifest["directory"], mode + ".log")
        if mode == "finish":
            answer, child = self.spawn_continuing(run_id, argv, log_path)
        else:
            answer, child = self.spawn(argv, log_path)
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


def check_points(points, declared):
    """A point list from a request, checked against the configuration.

    Returns the list to write, or an error object. Nothing here becomes a
    command-line argument: the file is written from names that were found in
    the configuration, and only its path is passed on.
    """
    if not isinstance(points, list) or len(points) == 0:
        return {"error": "points must be a non-empty list of coordinate sets"}
    if len(points) > POINT_LIMIT:
        return {"error": "at most {} points at a time".format(POINT_LIMIT)}

    checked = []
    for entry in points:
        if not isinstance(entry, dict) or len(entry) == 0:
            return {"error": "each point must be an object of dimension to values"}
        spec = dict()
        for dim, values in entry.items():
            if dim not in declared:
                return {"error": "unknown dimension '{}'".format(dim)}
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list) or len(values) == 0:
                return {"error": "no value chosen for {} in a point".format(dim)}
            values = [str(v) for v in values]
            known = declared[dim]
            if known is not None:
                # `*` is every value, and needs no checking against them
                unknown = [v for v in values if v != "*" and v not in known]
                if unknown:
                    return {
                        "error": "{} has no value {}".format(dim, ", ".join(unknown))
                    }
            spec[dim] = values
        checked.append(spec)
    return checked


def whoami():
    """`user@machine`, short and free.

    Worth saying on the page because the browser is often not on the machine
    doing the work: a forwarded port looks exactly like a local one, and the
    runs are the other host's. The short name rather than `ssh_target`'s fully
    qualified one — this is a label, not something to paste into ssh, and it
    costs no subprocess.
    """
    host = workspace.hostname()
    try:
        return "{}@{}".format(getpass.getuser(), host)
    except Exception:
        # no account to name: the host alone is still worth having
        return host


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
    # every stem this run has ever named, live or long finished — what a
    # capture-viewing request is checked against, so it can only ever open a
    # file this run itself chose to write
    stems = set()
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
        if kind == "run.started":
            # `finish` writes into this same file rather than a fresh one, so
            # a second attempt's progress must not be added on top of the
            # first's — this is where the count for the run as it stands now
            # starts over
            completed = 0
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
            trial_stem = record.get("stem")
            if trial_stem is not None:
                stems.add(trial_stem)
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
        "stems": stems,
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

    def __init__(self, root, base, port):
        super().__init__(("127.0.0.1", port), Handler)
        self.root = root
        # where the work is, which is not always where its record is kept:
        # --workspace separates them, and the configuration and the runs
        # started from here belong to the work
        self.base = base
        self.token = secrets.token_urlsafe(24)
        # runs started from here, so a second request cannot start them again
        self.finishing = dict()
        self.lock = threading.Lock()


# below this, a port belongs to the system and only root may listen on one
RESERVED_PORT = 1024
# a suggestion has to be a port somebody may actually have, and one that is
# not the first thing everything else on the machine tries
SUGGESTED_PORT = 8787
PORT_HINT = "pick a free one above {}, e.g. --port {}".format(
    RESERVED_PORT - 1, SUGGESTED_PORT
)


def bind(root, base, port):
    """Take the port, or say why it could not be had.

    Failing to listen is the whole command failing, and a traceback is a poor
    way to say so when the reason is one of a few knowable ones: the port is
    taken, it is one only root may have, or it is not a port at all.
    """
    if not isinstance(port, int) or not 0 <= port <= 65535:
        report(
            LogLevel.FATAL,
            "not a port number: {}".format(port),
            hint=["a port is between 0 and 65535", PORT_HINT],
        )
    try:
        return Server(root, base, port)
    except PermissionError:
        report(
            LogLevel.FATAL,
            "not allowed to listen on port {}".format(port),
            hint=[
                "ports below {} are reserved for root".format(RESERVED_PORT),
                PORT_HINT,
            ],
        )
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            existing = workspace.read_server(root)
            hints = ["something is already listening on {}".format(port)]
            if existing is not None and existing.get("port") == port:
                hints.append("it is the yuclid server with pid {}".format(existing["pid"]))
            hints.append(PORT_HINT)
            hints.append("or let one be chosen for you: yuclid serve")
            report(
                LogLevel.FATAL, "port {} is already in use".format(port), hint=hints
            )
        if e.errno == errno.EADDRNOTAVAIL:
            report(
                LogLevel.FATAL,
                "cannot listen on 127.0.0.1:{}".format(port),
                str(e),
                hint="the loopback interface is not available on this machine",
            )
        report(
            LogLevel.FATAL,
            "cannot listen on port {}".format(port),
            str(e),
            hint=PORT_HINT,
        )


def launch(args):
    # Serving an empty directory is useful: it can launch the first run from
    # the browser. Use the same workspace creation path as `yuclid run` rather
    # than requiring a run to have happened here already.
    root = workspace.open_root(args.directory, args.workspace)
    # everything this server is about comes from the workspace: the runs it
    # lists, the configuration it offers, and the directory a run it starts is
    # started in. Otherwise `--workspace` would move only half of it, and the
    # page would describe one place while starting runs in another.
    directory = workspace.work_of(root)
    if args.workspace is not None and args.directory is not None:
        report(
            LogLevel.WARNING,
            "--workspace decides where everything is, so {} is ignored".format(
                args.directory
            ),
        )
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

    server = bind(root, directory, args.port)
    if workspace.read_server(root) is None:
        # a forced second server does not take the note over: it stays with
        # the one that was here first, which is the one worth pointing at
        workspace.write_server(root, server.server_port)
    # so that `kill` leaves as tidily as Ctrl-C does
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    # the page carries the token itself, so the URL does not have to: a secret
    # in one ends up in scrollback, shell history and any Referer header
    url = "http://127.0.0.1:{}/".format(server.server_port)
    if not args.quiet:
        report(LogLevel.INFO, "watching", root)
        hints = ["stop with Ctrl-C"]
        # bound to loopback, so a machine reached over ssh needs the port
        # brought to where the browser is. Only worth saying when that is the
        # situation: sitting at the machine, the URL above is already the
        # whole story
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
        if not args.quiet:
            report(LogLevel.INFO, "stopped")
    finally:
        workspace.clear_server(root)
        server.server_close()

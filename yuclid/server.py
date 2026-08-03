"""A loopback view of the runs a directory holds.

It owns nothing. Every read comes from the run directories, which is why a run
that finished last week looks the same as one going now and why this can be
started, killed and started again while a run continues undisturbed. The one
thing it cannot do by reading is change a run, so control requests are forwarded
to the run's own socket and answered with what the run said it did.
"""

from yuclid.log import LogLevel, report
import yuclid.workspace as workspace
import yuclid.control as control
import http.server
import functools
import secrets
import json
import os
import re


PAGE = os.path.join(os.path.dirname(__file__), "web", "index.html")
BODY_LIMIT = 1 << 16


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
        match = re.fullmatch(r"/api/runs/([^/]+)/control", self.path.split("?")[0])
        if not match:
            return self.respond(404, {"error": "no such resource"})
        operation = self.body()
        if operation is None:
            return
        try:
            return self.respond(200, self.control(match.group(1), operation))
        except FileNotFoundError:
            return self.respond(404, {"error": "no such run"})

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
        return html.replace(b"__YUCLID_TOKEN__", self.server.token.encode())

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
        from yuclid.steer import progress_counts

        summaries = []
        for manifest in workspace.list_runs(self.server.root):
            counts = progress_counts(manifest["directory"])
            summaries.append(
                {
                    "id": manifest["id"],
                    "name": manifest.get("name"),
                    "state": manifest["state"],
                    "created": manifest.get("created"),
                    "output": manifest.get("output"),
                    "completed": counts[0] if counts else 0,
                    "total": counts[1] if counts else 0,
                }
            )
        return summaries

    def run(self, run_id):
        manifest = self.manifest(run_id)
        records = workspace.read_progress(manifest["directory"])

        # the last snapshot, not the first: the plan changes shape as points are
        # dropped and added, and one is written whenever it does
        plan = None
        completed, total, current, failed = 0, 0, None, False
        for record in records:
            if record["type"] == "plan":
                plan = record
            if record.get("total") is not None:
                total = record["total"]
            if record["type"] == "point.finished":
                completed += record.get("repetitions", 1)
                failed = failed or bool(record.get("failed"))
            if record["type"] == "point.started":
                current = record["key"]

        if plan is not None:
            failed = failed or any(p["status"] == "failed" for p in plan["points"])

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
            "dimensions": dimensions_of(plan),
            "completed": completed,
            "total": total,
            "current": current,
            "live": live,
            "paused": bool(plan.get("paused")) if plan and live else False,
            "failed": failed,
            "mood": mood(manifest["state"], plan, live, failed),
        }

    def progress(self, run_id, query):
        manifest = self.manifest(run_id)
        since = int(query.get("since") or 0)
        records = workspace.read_progress(manifest["directory"], since=since)
        return {"records": records, "seq": records[-1]["seq"] if records else since}

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


def dimensions_of(plan):
    """Each dimension's values, in the order the space puts them.

    A value counts as dropped once every point carrying it has been: that is
    what makes it something the page can offer to put back.
    """
    if plan is None:
        return {}
    seen = {dim: [] for dim in plan["order"]}
    dropped = {dim: {} for dim in plan["order"]}
    for point in plan["points"]:
        for dim, name in zip(plan["order"], point["key"]):
            if name not in seen[dim]:
                seen[dim].append(name)
                dropped[dim][name] = True
            if point["status"] != "dropped":
                dropped[dim][name] = False
    return {
        dim: [{"value": name, "dropped": dropped[dim][name]} for name in names]
        for dim, names in seen.items()
    }


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root, port):
        super().__init__(("127.0.0.1", port), Handler)
        self.root = root
        self.token = secrets.token_urlsafe(24)


def launch(args):
    root = workspace.find_root(args.directory)
    if root is None:
        report(
            LogLevel.FATAL,
            "no .yuclid directory in {}".format(
                os.path.abspath(args.directory or os.getcwd())
            ),
            hint="`yuclid run` creates one; name the directory holding it",
        )
    if not os.path.exists(PAGE):
        report(LogLevel.FATAL, "the web page is missing from the installation", PAGE)

    server = Server(root, args.port)
    url = "http://127.0.0.1:{}/?t={}".format(server.server_port, server.token)
    report(LogLevel.INFO, "watching", root)
    report(LogLevel.INFO, "open", url, hint="stop with Ctrl-C")
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        report(LogLevel.INFO, "stopped")
    finally:
        server.server_close()

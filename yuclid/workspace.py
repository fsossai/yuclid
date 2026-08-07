"""The per-run bookkeeping directory.

Every run writes `.yuclid/runs/<id>/`, whether or not anyone is watching. That is
what makes a run observable while it goes and reconstructable afterwards: the
progress file alone is enough to rebuild what a run did, so a reader that arrives
halfway through has missed nothing.

Reading a run's state is done by reading these files. Changing it needs an
authoritative answer from the process that owns the plan, which is what
`yuclid.control` is for.
"""

from yuclid.log import LogLevel, report
import itertools
import threading
import shutil
import socket
import json
import time
import os


DIRNAME = ".yuclid"
RUNS = "runs"
SERVER = "server.json"

MANIFEST = "manifest.json"
NAME = "name"
PROGRESS = "progress.jsonl"
CONTROL = "control.sock"
RESULTS = "results.jsonl"
POINTS = "points.json"
# point lists handed to a run on its command line live here rather than in the
# run's own directory: the command has to name a file that exists before the
# run does, and has to keep working once it has ended
POINT_SETS = "points"
TRIALS = "tmp"

RUNNING = "running"
FINISHED = "finished"
STOPPED = "stopped"
FAILED = "failed"
INTERRUPTED = "interrupted"


def hostname():
    return socket.gethostname().split(".")[0]


def workspace_of(chosen=None):
    """The workspace: where the configuration is, and where a run executes.

    Named explicitly with `--workspace`, or the working directory when it is
    not — the same rule every command follows, so the same invocation always
    means the same place. Its state directory is always `.yuclid` underneath
    it; there is no separate way to move that on its own.
    """
    return os.path.abspath(chosen) if chosen is not None else os.getcwd()


def root_path(workspace=None):
    """Where a workspace keeps its state: always `.yuclid` beneath it."""
    return os.path.join(workspace_of(workspace), DIRNAME)


def find_root(workspace=None):
    """The state directory of a workspace, or None.

    Looked up in one directory and not searched for upwards, the same rule the
    configuration follows. Walking up would be worse than inconsistent here:
    `yuclid tplot` already keeps a cache in `~/.yuclid`, so every run made
    anywhere below a home directory would be recorded into it.
    """
    root = root_path(workspace)
    return root if os.path.isdir(root) else None


def open_root(workspace=None):
    """As `find_root`, creating the directory when there is none."""
    root = root_path(workspace)
    os.makedirs(os.path.join(root, RUNS), exist_ok=True)
    return root


def create_run(root, stamp, **manifest):
    """Claim a directory for this run, and say what run it is.

    The timestamp alone collides when a job array starts thirty-two tasks in the
    same second, so the name is claimed by an exclusive `mkdir` and a suffix is
    tried until one succeeds. That settles the race between hosts sharing a
    filesystem without carrying a hostname and a pid around in every path.
    """
    base = os.path.join(root, RUNS)
    for attempt in itertools.count(1):
        run_id = stamp if attempt == 1 else "{}-{}".format(stamp, attempt)
        directory = os.path.join(base, run_id)
        try:
            os.makedirs(directory)
        except FileExistsError:
            continue
        break

    os.makedirs(os.path.join(directory, TRIALS), exist_ok=True)
    manifest.update(
        {
            "id": run_id,
            "pid": os.getpid(),
            "host": hostname(),
            "created": time.time(),
            "state": RUNNING,
        }
    )
    write_manifest(directory, manifest)
    return run_id, directory


def reopen_run(root, run_id):
    """Resume an existing run's directory instead of starting a new one.

    `finish` fills the gaps a run left behind, and the record of that belongs
    to the run whose gaps they are: continuing it, rather than starting a
    fresh run that happens to write into the same output file. The pid and
    host are updated to this process, the one now actually doing the work, so
    that a reader checking whether the run is alive is not looking at the pid
    of a process that has long since exited.
    """
    directory = run_directory(root, run_id)
    manifest = read_manifest(directory)
    if manifest is None:
        raise FileNotFoundError(run_id)
    manifest["pid"] = os.getpid()
    manifest["host"] = hostname()
    manifest["state"] = RUNNING
    manifest.pop("ended", None)
    write_manifest(directory, manifest)
    os.makedirs(os.path.join(directory, TRIALS), exist_ok=True)
    return run_id, directory


def write_manifest(directory, manifest):
    path = os.path.join(directory, MANIFEST)
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    # a reader must never catch a half-written manifest
    os.replace(temporary, path)


def read_manifest(directory):
    try:
        with open(os.path.join(directory, MANIFEST)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def set_state(directory, state):
    manifest = read_manifest(directory)
    if manifest is None:
        return
    manifest["state"] = state
    manifest["ended"] = time.time()
    write_manifest(directory, manifest)


def is_alive(manifest):
    """Whether the process that wrote this manifest is still running here.

    A run on another host cannot be inspected and cannot be steered either, so
    it is reported as not alive rather than guessed about.
    """
    if manifest.get("host") != hostname():
        return False
    try:
        os.kill(manifest["pid"], 0)
    except (OSError, TypeError, KeyError):
        return False
    return True


def state_of(manifest):
    """The recorded state, corrected for a run whose process is gone.

    Nothing watches a run, so an interruption is noticed by whoever reads the
    manifest next rather than recorded when it happens.
    """
    state = manifest.get("state", INTERRUPTED)
    if state == RUNNING and not is_alive(manifest):
        return INTERRUPTED
    return state


def run_directory(root, run_id):
    return os.path.join(root, RUNS, run_id)


def list_runs(root):
    """Every run the directory holds, newest first."""
    base = os.path.join(root, RUNS)
    if not os.path.isdir(base):
        return []
    runs = []
    for name in sorted(os.listdir(base), reverse=True):
        directory = os.path.join(base, name)
        manifest = read_manifest(directory)
        if manifest is None:
            continue
        manifest["state"] = state_of(manifest)
        manifest["directory"] = directory
        manifest["name"] = read_name(directory)
        runs.append(manifest)
    return runs


def live_runs(root):
    return [m for m in list_runs(root) if m["state"] == RUNNING]


def delete_run(root, run_id):
    """Remove a run's directory, and only ever a run's directory.

    The results themselves survive: the file in the working directory and the
    one here are two names for one inode, so taking this one away leaves the
    measurements where the user can still see them. What goes is the
    bookkeeping — the manifest, the progress, the trial captures.
    """
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    if run_id in ("", ".", "..") or any(s in run_id for s in separators):
        raise ValueError("not a run name: {}".format(run_id))
    directory = run_directory(root, run_id)
    if read_manifest(directory) is None:
        raise FileNotFoundError(run_id)
    shutil.rmtree(directory)


def clear_temporary(root, run_id):
    """Empty a run's scratch directories, keeping the run itself.

    What a trial and a setup command printed is the bulk of a run directory and
    the least of its meaning: the manifest, the progress and the plan are what
    say the run happened, and they stay. The directories stay too, so that
    finishing the run later has somewhere to write.

    A hard link is left alone and not counted: the results file is one, and it
    is not this directory's to give back.
    """
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    if run_id in ("", ".", "..") or any(s in run_id for s in separators):
        raise ValueError("not a run name: {}".format(run_id))
    directory = run_directory(root, run_id)
    if read_manifest(directory) is None:
        raise FileNotFoundError(run_id)

    freed = 0
    for name in (TRIALS, "setup"):
        scratch = os.path.join(directory, name)
        if not os.path.isdir(scratch):
            continue
        for entry in os.listdir(scratch):
            path = os.path.join(scratch, entry)
            try:
                stat = os.lstat(path)
                if os.path.isdir(path) and not os.path.islink(path):
                    freed += directory_size(path)
                    shutil.rmtree(path)
                    continue
                if stat.st_nlink > 1:
                    continue
                freed += stat.st_size
                os.unlink(path)
            except OSError:
                pass
    return freed


def directory_size(path):
    """The room a tree takes, counting a hard-linked file as nothing."""
    total = 0
    for where, _, files in os.walk(path):
        for name in files:
            try:
                stat = os.lstat(os.path.join(where, name))
                total += 0 if stat.st_nlink > 1 else stat.st_size
            except OSError:
                pass
    return total


def temporary_size(directory):
    """The room a run's scratch directories take."""
    return sum(
        directory_size(os.path.join(directory, name))
        for name in (TRIALS, "setup")
        if os.path.isdir(os.path.join(directory, name))
    )


def write_point_set(root, stamp, points, replay_of=None):
    """Keep a point list where a command line can name it, and say where.

    The file outlives the run that was started from it, so the run's recorded
    command stays runnable: finishing or replaying it later reads the same
    list rather than a path that has gone.
    """
    directory = os.path.join(root, POINT_SETS)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "{}.json".format(stamp))
    body = {"points": points}
    if replay_of is not None:
        body["replay_of"] = replay_of
    with open(path, "w") as f:
        json.dump(body, f, indent=2)
        f.write("\n")
    return path


def read_server(root):
    """The server watching this directory, if one still is.

    A courtesy note left by `yuclid serve`, not a claim on anything: a server
    owns no run, so a stale file from one that was killed is simply ignored.
    """
    try:
        with open(os.path.join(root, SERVER)) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if info.get("host") != hostname():
        return None
    try:
        os.kill(info["pid"], 0)
    except (OSError, TypeError, KeyError):
        return None
    return info


def write_server(root, port):
    info = {
        "pid": os.getpid(),
        "host": hostname(),
        "port": port,
        "started": time.time(),
    }
    with open(os.path.join(root, SERVER), "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return info


def clear_server(root):
    """Take the note down, unless it was another server that left it."""
    path = os.path.join(root, SERVER)
    try:
        with open(path) as f:
            if json.load(f).get("pid") != os.getpid():
                return
        os.unlink(path)
    except (OSError, ValueError):
        pass


def run_of_output(root, path):
    """The run that wrote this result file, if this directory holds it.

    Matched by path: a run records the destination it keeps its results.jsonl
    copied to, and that is the name this checks against, whichever run left it.
    """
    absolute = os.path.abspath(path)
    for manifest in list_runs(root):
        if manifest.get("output") == absolute:
            return manifest
    return None


def select_run(root, run_id=None):
    """The run a steering command is addressed to.

    A verb with an invisible object is only safe while there is exactly one
    candidate, so anything else asks for `--run`.
    """
    if run_id is not None:
        directory = run_directory(root, run_id)
        manifest = read_manifest(directory)
        if manifest is None:
            report(LogLevel.FATAL, "no such run", run_id)
        manifest["state"] = state_of(manifest)
        manifest["directory"] = directory
        manifest["name"] = read_name(directory)
        return manifest

    live = live_runs(root)
    if len(live) == 0:
        report(
            LogLevel.FATAL,
            "no run is in progress",
            hint="`yuclid runs` lists the ones that finished",
        )
    if len(live) > 1:
        report(
            LogLevel.FATAL,
            "{} runs are in progress".format(len(live)),
            hint="name one with --run {}".format(live[0]["id"]),
        )
    return live[0]


def read_name(directory):
    """The name given to a run, or None."""
    try:
        with open(os.path.join(directory, NAME)) as f:
            return f.read().strip() or None
    except OSError:
        return None


def check_name(name):
    """The name, stripped, or a ValueError saying why it is not one."""
    name = (name or "").strip()
    if len(name) > 120 or "\n" in name or "\r" in name:
        raise ValueError("a name is one line of at most 120 characters")
    return name


def write_name(directory, name):
    """Name a run, or clear its name with an empty one.

    Kept in a file of its own rather than in the manifest, which the run writes
    and nothing else does. A rename arriving while the run records its final
    state would otherwise be a lost update between two read-modify-writes.
    """
    name = check_name(name)
    path = os.path.join(directory, NAME)
    if name == "":
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    with open(path, "w") as f:
        f.write(name + "\n")
    return name


def results_path(directory):
    """Where a run keeps what it has measured so far.

    Always this same name, in the run's own directory: every trial appends
    here directly, so it is the one file that is never behind what the run
    has actually done, whatever `--output` eventually asks to be copied to.
    """
    return os.path.join(directory, RESULTS)


class Progress:
    """What the run did, one JSON object per line, written by the run alone.

    The first record lists every point of the plan, so a reader that arrives
    late can rebuild the whole picture from this file without having watched.
    """

    def __init__(self, path):
        self.lock = threading.Lock()
        # `finish` reopens an existing run and appends to its progress file:
        # starting back at 0 would write records a client already holds a
        # higher seq for, and its incremental /progress polling would never
        # see them
        self.seq = last_seq(path) if path is not None else 0
        # compiling a script is not a run and has nothing to record
        self.stream = open(path, "a") if path is not None else None

    def emit(self, kind, **fields):
        if self.stream is None:
            return
        with self.lock:
            self.seq += 1
            record = {"seq": self.seq, "time": time.time(), "type": kind}
            record.update(fields)
            self.stream.write(json.dumps(record, default=str) + "\n")
            self.stream.flush()

    def close(self):
        if self.stream is None:
            return
        with self.lock:
            self.stream.close()
            self.stream = None


def last_seq(path):
    """The highest seq already written to a progress file, or 0.

    What a fresh `Progress` continuing that file must count up from, so its
    records keep the sequence strictly increasing across the reopen.
    """
    seq = 0
    try:
        with open(path) as f:
            for line in f:
                if not line.endswith("\n"):
                    break
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                seq = max(seq, record.get("seq", 0))
    except OSError:
        pass
    return seq


def read_progress(directory, since=0):
    """The progress records after `since`.

    A trailing partial line is skipped rather than reported: the run may be
    writing one at this very moment, and it will be complete by the next read.
    """
    path = os.path.join(directory, PROGRESS)
    records = []
    try:
        with open(path) as f:
            for line in f:
                if not line.endswith("\n"):
                    break
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("seq", 0) > since:
                    records.append(record)
    except OSError:
        pass
    return records

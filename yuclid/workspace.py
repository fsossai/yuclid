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
import socket
import errno
import json
import time
import os


DIRNAME = ".yuclid"
RUNS = "runs"

MANIFEST = "manifest.json"
NAME = "name"
PROGRESS = "progress.jsonl"
CONTROL = "control.sock"
RESULTS = "results.yuclid.jsonl"
REPLAY = "replay.json"
TRIALS = "tmp"

RUNNING = "running"
FINISHED = "finished"
STOPPED = "stopped"
FAILED = "failed"
INTERRUPTED = "interrupted"


def hostname():
    return socket.gethostname().split(".")[0]


def root_path(start=None):
    return os.path.join(os.path.abspath(start or os.getcwd()), DIRNAME)


def find_root(start=None):
    """The `.yuclid` directory of `start`, or None.

    Looked up in one directory and not searched for upwards, the same rule the
    configuration follows. Walking up would be worse than inconsistent here:
    `yuclid tplot` already keeps a cache in `~/.yuclid`, so every run made
    anywhere below a home directory would be recorded into it.
    """
    root = root_path(start)
    return root if os.path.isdir(root) else None


def open_root(start=None):
    """As `find_root`, creating the directory when there is none."""
    root = root_path(start)
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


def run_of_output(root, path):
    """The run that wrote this result file, if this directory holds it.

    Matched by inode: the run directory holds a hard link to the very file, so
    the two names are the same file however either of them was reached. A run
    whose output landed on another filesystem is matched by path instead, since
    there it is a symlink rather than a link.
    """
    try:
        wanted = os.stat(path).st_ino
    except OSError:
        return None
    absolute = os.path.abspath(path)
    for manifest in list_runs(root):
        try:
            if os.stat(os.path.join(manifest["directory"], RESULTS)).st_ino == wanted:
                return manifest
        except OSError:
            pass
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


def link_results(directory, output):
    """A second name, in the run directory, for the file the run writes.

    A hard link rather than a copy: both names see the records as they are
    appended, neither is a duplicate, and the run keeps its data once the
    working directory has been tidied. Across filesystems there is no such link,
    so a symlink stands in and the manifest records where the file really is.
    """
    target = os.path.join(directory, RESULTS)
    if os.path.exists(target):
        return
    try:
        os.link(output, target)
        return
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
            report(LogLevel.WARNING, "cannot keep the results in the run directory", str(e))
            return
    try:
        os.symlink(os.path.abspath(output), target)
    except OSError as e:
        report(LogLevel.WARNING, "cannot keep the results in the run directory", str(e))


class Progress:
    """What the run did, one JSON object per line, written by the run alone.

    The first record lists every point of the plan, so a reader that arrives
    late can rebuild the whole picture from this file without having watched.
    """

    def __init__(self, path):
        self.lock = threading.Lock()
        # compiling a script is not a run and has nothing to record
        self.stream = open(path, "a") if path is not None else None
        self.seq = 0

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

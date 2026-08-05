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
import errno
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
RESULTS = "results.yuclid.jsonl"
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


def root_path(start=None, workspace=None):
    """Where this invocation keeps its state.

    Normally `.yuclid` beside the work, which is what makes a directory of
    experiments self-contained. `workspace` names the directory itself instead,
    for the times when the work and the record of it cannot live together: a
    read-only checkout, a scratch filesystem worth more than the home one, or
    several working directories reporting into one place.
    """
    if workspace is not None:
        return os.path.abspath(workspace)
    return os.path.join(os.path.abspath(start or os.getcwd()), DIRNAME)


def find_root(start=None, workspace=None):
    """The state directory of `start`, or None.

    Looked up in one directory and not searched for upwards, the same rule the
    configuration follows. Walking up would be worse than inconsistent here:
    `yuclid tplot` already keeps a cache in `~/.yuclid`, so every run made
    anywhere below a home directory would be recorded into it.
    """
    root = root_path(start, workspace)
    return root if os.path.isdir(root) else None


def open_root(start=None, workspace=None):
    """As `find_root`, creating the directory when there is none."""
    root = root_path(start, workspace)
    os.makedirs(os.path.join(root, RUNS), exist_ok=True)
    return root


def work_of(root):
    """The directory a workspace is about.

    A workspace sitting beside the work is named `.yuclid`, and the work is the
    directory holding it. One put somewhere else is the whole of it: the
    configuration, the runs and everything they wrote are in the one place, so
    that a workspace can be moved, copied or served without carrying a second
    path around.
    """
    return os.path.dirname(root) if os.path.basename(root) == DIRNAME else root


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

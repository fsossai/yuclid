"""What a run still intends to do, and the operations that change it.

`prepare_subspace_execution` decides the points once; this holds them as
something that can still be altered while they are being executed. Every method
takes the lock, because points may be running in parallel and a steering
operation may arrive at any moment.

An operation returns what it did rather than merely succeeding, since the caller
is a person waiting to be told. What comes back is a decision, though, not always
a completed effect: dropping points says they will not run, and pausing says the
flag is set, which is not the same as the run being idle.
"""

from yuclid.control import ControlError
import threading


PENDING = "pending"
RUNNING = "running"
DONE = "done"
DROPPED = "dropped"
KILLED = "killed"
FAILED = "failed"

FINISHED = {DONE, DROPPED, KILLED, FAILED}


class Plan:
    def __init__(self, order, points, repeat, recorded=None, expand=None, undefined=None):
        self.lock = threading.RLock()
        self.order = list(order)
        self.repeat = repeat
        self.expand = expand
        # These dimensions had no domain in the configuration. Their values
        # came from a selector or preset, so they are the ones the web UI can
        # sensibly invite someone to extend while this run is under way.
        self.undefined = list(undefined or [])
        self.stopped = False
        self.paused = False
        self.resumed = threading.Event()
        self.resumed.set()
        self.entries = []
        for point in points:
            self.entries.append(self._entry(point, recorded))
        self._renumber()

    def _entry(self, point, recorded=None):
        key = tuple(str(x["name"]) for x in point)
        done = 0
        if recorded is not None:
            done = min(recorded.get(key, 0), self.repeat)
        return {
            "seq": 0,
            "key": key,
            "point": point,
            "target": self.repeat,
            "done": done,
            "status": DONE if done >= self.repeat else PENDING,
            "resumed": done,
        }

    def _renumber(self):
        for i, entry in enumerate(self.entries, start=1):
            entry["seq"] = i

    def conform(self, wanted):
        """Cut this plan down to what a previous run measured.

        A replay is that run again: the points it ran, each for as many
        repetitions as it managed, and the ones it skipped skipped again.
        Everything else is dropped — a point the original never reached is not
        part of what it did, whether it was dropped, killed, or simply left
        when the run was stopped.

        Returns the keys it recognised, so a caller covering several presets
        can tell which of them nothing here accounts for.
        """
        with self.lock:
            # A previous run may have acquired a value through live steering.
            # It is absent from the source configuration that built this plan,
            # so make its exact point before deciding what to keep. Naming
            # every coordinate prevents an added value from expanding into
            # unrelated points; `expand` still applies its setup and
            # conditions just as it did the first time.
            known = {entry["key"] for entry in self.entries}
            for key in wanted:
                if key not in known and self.expand is not None:
                    self._add(
                        {
                            "coords": {
                                dim: [value] for dim, value in zip(self.order, key)
                            }
                        }
                    )
                    known = {entry["key"] for entry in self.entries}

            matched = set()
            for entry in self.entries:
                want = wanted.get(entry["key"])
                if want is None:
                    entry["status"] = DROPPED
                    continue
                matched.add(entry["key"])
                entry["target"] = want["repetitions"]
                if want.get("skipped"):
                    # skipped is not measured: it is counted as done and left
                    # alone, which is what the original run did with it
                    entry["resumed"] = entry["target"]
                    entry["done"] = entry["target"]
                if entry["done"] >= entry["target"]:
                    entry["status"] = DONE
            return matched

    # -- what the run asks -------------------------------------------------

    def skipped(self):
        """The points a previous run already recorded in full."""
        with self.lock:
            return [e for e in self.entries if e["resumed"] >= e["target"]]

    def pending(self):
        """Yield the next point to run, until there is none or the run stops.

        Blocks while paused rather than spinning, and ends after `stop` without
        interrupting what is already in flight.
        """
        while True:
            self.resumed.wait()
            with self.lock:
                if self.stopped:
                    return
                entry = next(
                    (
                        e
                        for e in self.entries
                        if e["status"] == PENDING and e["done"] < e["target"]
                    ),
                    None,
                )
                if entry is None:
                    return
                entry["status"] = RUNNING
            yield entry

    def finish(self, entry, status=DONE):
        with self.lock:
            entry["status"] = status

    def hold(self, entry):
        """Put a point back in the queue with the repetitions it managed.

        A pause stops the run between one repetition and the next, so a point
        may be left part measured. It is pending again, not finished: resuming
        carries on from where it stopped rather than starting it over.
        """
        with self.lock:
            entry["status"] = PENDING

    def repetition_done(self, entry):
        with self.lock:
            entry["done"] += 1

    def abandon_repetition(self, entry):
        """An abandoned repetition is one fewer, not one to try again.

        Without this the point would keep starting the repetition that was just
        killed, which is the opposite of what killing it meant.
        """
        with self.lock:
            entry["target"] = max(entry["done"], entry["target"] - 1)

    def units(self):
        """(total, completed), counted in repetitions as the progress line is."""
        with self.lock:
            total = sum(e["target"] for e in self.entries if e["status"] != DROPPED)
            return total, sum(e["done"] for e in self.entries)

    def units_before(self, entry):
        """The repetitions belonging to the points ahead of this one."""
        with self.lock:
            return sum(
                e["target"]
                for e in self.entries
                if e["seq"] < entry["seq"] and e["status"] != DROPPED
            )

    def width(self):
        with self.lock:
            return len(str(len(self.entries)))

    def running(self):
        with self.lock:
            return [e for e in self.entries if e["status"] == RUNNING]

    def in_flight(self, key):
        """The entry with this key that is executing, if one is."""
        with self.lock:
            return next(
                (e for e in self.entries if e["key"] == key and e["status"] == RUNNING),
                None,
            )

    def snapshot(self):
        """The whole plan as it stands, which is what a late reader needs.

        Written at the start and again after every steering operation, so the
        last one in the progress file is always the current picture.
        """
        with self.lock:
            return {
                "order": list(self.order),
                "undefined": list(self.undefined),
                "paused": self.paused,
                "total": sum(
                    e["target"] for e in self.entries if e["status"] != DROPPED
                ),
                "points": [
                    {
                        "seq": e["seq"],
                        "key": list(e["key"]),
                        "target": e["target"],
                        "done": e["done"],
                        "status": e["status"],
                    }
                    for e in self.entries
                ],
            }

    # -- what a steering command asks -------------------------------------

    def apply(self, op):
        kind = op.get("op")
        handler = {
            "pause": self._pause,
            "resume": self._resume,
            "stop": self._stop,
            "drop": self._drop,
            "add": self._add,
            "repeat": self._repeat,
            "order": self._order,
        }.get(kind)
        if handler is None:
            raise ControlError("unknown operation '{}'".format(kind))
        with self.lock:
            return handler(op)

    def _state(self, **fields):
        total, completed = (
            sum(e["target"] for e in self.entries if e["status"] != DROPPED),
            sum(e["done"] for e in self.entries),
        )
        fields.update({"completed": completed, "total": total})
        return fields

    def _pause(self, op):
        self.paused = True
        self.resumed.clear()
        return self._state(paused=True, in_flight=len(self.running()))

    def _resume(self, op):
        self.paused = False
        self.resumed.set()
        return self._state(paused=False)

    def _stop(self, op):
        self.stopped = True
        # a stop while paused has to wake the loop up to let it end
        self.resumed.set()
        remaining = [e for e in self.entries if e["status"] == PENDING]
        # what is in flight is killed by the caller, which is the only one that
        # can reach the commands; here it is enough that they are counted
        return self._state(
            stopped=True,
            abandoned=len(remaining),
            interrupted=len(self.running()),
        )

    def _matching(self, op):
        """The coordinates an operation names, checked against this space.

        A dimension named with no values means every value it has: naming the
        dimension is the whole of what was meant.
        """
        coords = op.get("coords") or {}
        if not isinstance(coords, dict) or len(coords) == 0:
            raise ControlError("name at least one dimension, as dim=value")
        unknown = [dim for dim in coords if dim not in self.order]
        if unknown:
            raise ControlError(
                "unknown dimension(s) {}: this run has {}".format(
                    ", ".join(unknown), ", ".join(self.order)
                )
            )
        index = {dim: self.order.index(dim) for dim in coords}
        return coords, index

    def _drop(self, op):
        coords, index = self._matching(op)

        def matches(entry):
            return all(
                not values or entry["key"][index[dim]] in values
                for dim, values in coords.items()
            )

        dropped = 0
        for entry in self.entries:
            if entry["status"] == PENDING and matches(entry):
                entry["status"] = DROPPED
                dropped += 1
        return self._state(dropped=dropped)

    def _add(self, op):
        coords, _ = self._matching(op)
        if self.expand is None:
            raise ControlError("this run cannot add points")
        known = {e["key"] for e in self.entries}
        added, restored = 0, 0

        for entry in self.entries:
            if entry["status"] == DROPPED and self._covers(coords, entry):
                entry["status"] = PENDING
                restored += 1

        for point in self.expand(coords):
            entry = self._entry(point)
            if entry["key"] in known:
                continue
            known.add(entry["key"])
            self.entries.append(entry)
            added += 1

        self._renumber()
        return self._state(added=added, restored=restored)

    def _covers(self, coords, entry):
        return all(
            not values or entry["key"][self.order.index(dim)] in values
            for dim, values in coords.items()
        )

    def _repeat(self, op):
        value = op.get("value")
        if not isinstance(value, int) or value < 1:
            raise ControlError("repeat needs a positive number of repetitions")
        changed = 0
        for entry in self.entries:
            if entry["status"] in (PENDING, RUNNING):
                entry["target"] = value
                changed += 1
                if entry["done"] >= value:
                    entry["status"] = DONE
        return self._state(retargeted=changed, repeat=value)

    def _order(self, op):
        dims = op.get("order") or []
        unknown = [d for d in dims if d not in self.order]
        if unknown:
            raise ControlError(
                "unknown dimension(s) {}: this run has {}".format(
                    ", ".join(unknown), ", ".join(self.order)
                )
            )
        # a value's position in the space, not its spelling, decides the order
        positions = {dim: {} for dim in self.order}
        for entry in self.entries:
            for dim, name in zip(self.order, entry["key"]):
                positions[dim].setdefault(name, len(positions[dim]))

        rest = [d for d in self.order if d not in dims]
        ranking = list(dims) + rest

        def rank(entry):
            return tuple(
                positions[dim][entry["key"][self.order.index(dim)]] for dim in ranking
            )

        head = [e for e in self.entries if e["status"] != PENDING]
        tail = sorted((e for e in self.entries if e["status"] == PENDING), key=rank)
        self.entries = head + tail
        self._renumber()
        return self._state(reordered=len(tail), order=ranking)

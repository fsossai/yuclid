"""One JSON request, one JSON response, over a Unix socket in the run directory.

Reading what a run is doing is done by reading its files, which works whether it
is still going or finished last week. Changing what it does needs an
authoritative answer from the one process that owns the plan — whether the
operation was valid, and what it affected — so that is a call rather than a
message left somewhere.

The socket carries no credentials because it does not need any: the file it lives
at is created with mode 0600, so the permission to steer a run is the permission
to read its directory.
"""

import threading
import socket
import json
import os


class ControlError(Exception):
    """An operation the run declines, reported back to whoever asked."""


def read_message(stream):
    """One newline-terminated JSON object, or None if the peer said nothing."""
    buffer = b""
    while not buffer.endswith(b"\n"):
        chunk = stream.recv(4096)
        if not chunk:
            break
        buffer += chunk
        if len(buffer) > 1 << 20:
            raise ControlError("message too long")
    if not buffer.strip():
        return None
    return json.loads(buffer.decode())


def write_message(stream, message):
    stream.sendall(json.dumps(message).encode() + b"\n")


class Server:
    """Answers control requests for as long as the run lasts.

    Binding is best effort: a run must never be lost because its directory sits
    on a filesystem without Unix sockets, or because the path came out longer
    than the platform allows.
    """

    def __init__(self, path, handler):
        self.path = path
        self.handler = handler
        self.socket = None
        self.thread = None
        self.closed = False

    def start(self):
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.bind(self.path)
            os.chmod(self.path, 0o600)
            self.socket.listen(8)
        except (OSError, AttributeError) as e:
            self.socket = None
            return str(e)
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()
        return None

    def _accept(self):
        while not self.closed:
            try:
                connection, _ = self.socket.accept()
            except OSError:
                return
            with connection:
                self._answer(connection)

    def _answer(self, connection):
        try:
            message = read_message(connection)
            if message is None:
                return
            response = self.handler(message)
            write_message(connection, {"ok": True, "result": response or {}})
        except ControlError as e:
            write_message(connection, {"ok": False, "error": str(e)})
        except Exception as e:
            # a bad request must not take the run down with it
            try:
                write_message(connection, {"ok": False, "error": repr(e)})
            except OSError:
                pass

    def close(self):
        self.closed = True
        if self.socket is not None:
            try:
                self.socket.close()
            finally:
                self.socket = None
        try:
            os.unlink(self.path)
        except OSError:
            pass


def request(path, message, timeout=10.0):
    """Send one operation to a run and return what it says it did."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        write_message(s, message)
        response = read_message(s)
    if response is None:
        raise ControlError("the run closed the connection without answering")
    if not response.get("ok"):
        raise ControlError(response.get("error", "the run declined the operation"))
    return response.get("result", {})

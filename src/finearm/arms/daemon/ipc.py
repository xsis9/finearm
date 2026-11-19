"""Unix socket JSON-lines IPC for the arm control daemon (arm-neutral).

Dimension checks (snapshot 14/7, policy action 7) are parameterized; the
module defaults keep the legacy shipped-arm (panthera_ht) dimensions so
existing no-argument call sites behave exactly as before. The ``status``
response additionally carries the daemon's profile handshake (additive,
backward compatible).
"""

from __future__ import annotations

import errno
import json
import math
import os
import socket
import stat
import threading
import time

# Legacy defaults = the shipped panthera_ht profile (6 joints + gripper).
DEFAULT_STATE_DIM = 14
DEFAULT_ACTION_DIM = 7


class ControlClientError(RuntimeError):
    pass


class DaemonDisconnectedError(ControlClientError):
    pass


class MalformedResponseError(ControlClientError):
    pass


class StaleSnapshotError(ControlClientError):
    pass


class ControlServer:
    def __init__(self, socket_path: str, daemon, read_timeout_s: float = 1.0, max_request_bytes: int = 65536):
        self.socket_path = socket_path
        self.daemon = daemon
        self.read_timeout_s = float(read_timeout_s)
        self.max_request_bytes = int(max_request_bytes)
        self._sock = None
        self._thread = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._sock or self._thread:
            self.stop()
        self._prepare_socket_path()
        self._stop_event.clear()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.socket_path)
        self._sock.listen(16)
        self._sock.settimeout(0.1)
        self._thread = threading.Thread(target=self._serve, name="finearm-control-ipc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            thread.start()

    def _handle_client(self, conn) -> None:
        with conn:
            conn.settimeout(self.read_timeout_s)
            try:
                request = _read_json_line(conn, max_bytes=self.max_request_bytes)
                response = self._dispatch(request)
            except socket.timeout:
                response = {"ok": False, "error": "request read timed out"}
            except Exception as exc:
                response = {"ok": False, "error": repr(exc)}
            try:
                conn.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8"))
            except OSError:
                pass

    def _prepare_socket_path(self) -> None:
        try:
            mode = os.stat(self.socket_path).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RuntimeError(f"socket path exists and is not a socket: {self.socket_path}")
        if _is_active_socket(self.socket_path):
            raise RuntimeError(f"socket path is already served by an active daemon: {self.socket_path}")
        os.unlink(self.socket_path)

    def _dispatch(self, request: dict) -> dict:
        command = request.get("request")
        if command == "status":
            status = dict(self.daemon.status())
            # Profile handshake (Phase 2, additive): lets clients learn the
            # arm's dimensions and feature names. Old clients ignore it.
            status["profile"] = self.daemon.profile_handshake()
            return {"ok": True, "type": "status", "status": status}
        if command == "snapshot":
            return {"ok": True, "type": "snapshot", "snapshot": self.daemon.latest_snapshot()}
        if command == "hold":
            self.daemon.hold()
            return {"ok": True, "type": "hold", "status": self.daemon.status()}
        if command == "set_mode":
            self.daemon.set_control_mode(request.get("mode"))
            return {"ok": True, "type": "set_mode", "status": self.daemon.status()}
        if command == "policy_action":
            self.daemon.submit_policy_action(request.get("action"))
            return {"ok": True, "type": "policy_action", "status": self.daemon.status()}
        if command == "stop":
            threading.Thread(target=self.daemon.stop, daemon=True).start()
            return {"ok": True, "type": "stop"}
        return {"ok": False, "error": f"unknown request: {command!r}"}


def _is_active_socket(socket_path: str) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.05)
        probe.connect(socket_path)
        return True
    except OSError as exc:
        if exc.errno in {errno.ECONNREFUSED, errno.ENOENT, errno.ENOTSOCK}:
            return False
        return False
    finally:
        probe.close()


class ControlClient:
    def __init__(
        self,
        socket_path: str,
        timeout_s: float = 1.0,
        *,
        state_dim: int = DEFAULT_STATE_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
    ):
        self.socket_path = socket_path
        self.timeout_s = float(timeout_s)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

    @classmethod
    def from_profile(cls, profile, socket_path: str | None = None, timeout_s: float = 1.0) -> "ControlClient":
        """Build a client whose validation dims come from an ArmProfile."""
        return cls(
            socket_path if socket_path is not None else profile.socket_path,
            timeout_s,
            state_dim=profile.state_dim,
            action_dim=profile.action_dim,
        )

    def status(self) -> dict:
        response = self._request("status")
        return _require_payload(response, "status")

    def snapshot(self, max_age_s: float | None = None) -> dict:
        response = self._request("snapshot")
        snapshot = _require_payload(response, "snapshot")
        _validate_snapshot(snapshot, self.state_dim, self.action_dim)
        if max_age_s is not None and time.monotonic() - float(snapshot["timestamp"]) > float(max_age_s):
            raise StaleSnapshotError("snapshot is stale")
        return snapshot

    def hold(self) -> dict:
        response = self._request("hold")
        return _require_payload(response, "status")

    def set_mode(self, mode: str) -> dict:
        if mode not in {"teleop", "policy", "hold", "freedrive"}:
            raise ValueError(f"unknown control mode: {mode!r}")
        response = self._request("set_mode", mode=mode)
        return _require_payload(response, "status")

    def send_policy_action(self, action) -> dict:
        response = self._request("policy_action", action=_action_as_floats(action, self.action_dim))
        return _require_payload(response, "status")

    def stop(self) -> bool:
        response = self._request("stop")
        return bool(response.get("ok"))

    def _request(self, command: str, **payload) -> dict:
        request = {"request": command}
        request.update(payload)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout_s)
                sock.connect(self.socket_path)
                sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
                raw = _read_line(sock)
        except DaemonDisconnectedError:
            raise
        except OSError as exc:
            raise DaemonDisconnectedError(str(exc)) from exc
        try:
            response = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise MalformedResponseError("daemon returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise MalformedResponseError("daemon response must be a JSON object")
        if not response.get("ok", False):
            raise MalformedResponseError(str(response.get("error", "daemon returned an error")))
        return response


def _read_line(sock, max_bytes: int | None = None) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            if not chunks:
                raise DaemonDisconnectedError("daemon closed the connection")
            break
        chunks.append(chunk)
        total = len(b"".join(chunks))
        if max_bytes is not None and total > max_bytes:
            raise ValueError("request line too large")
        if b"\n" in chunk:
            break
    return b"".join(chunks).split(b"\n", 1)[0]


def _read_json_line(sock, max_bytes: int | None = None) -> dict:
    raw = _read_line(sock, max_bytes=max_bytes)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    return value


def _require_payload(response: dict, key: str):
    if key not in response:
        raise MalformedResponseError(f"daemon response missing {key!r}")
    return response[key]


def _action_as_floats(action, action_dim: int = DEFAULT_ACTION_DIM) -> list[float]:
    if not isinstance(action, (list, tuple)) or len(action) != action_dim:
        raise ValueError(f"policy action must contain {action_dim} values")
    result = []
    for value in action:
        if isinstance(value, bool):
            raise ValueError("policy action values must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("policy action values must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError("policy action values must be finite")
        result.append(numeric)
    return result


def _validate_snapshot(snapshot: dict, state_dim: int = DEFAULT_STATE_DIM, action_dim: int = DEFAULT_ACTION_DIM) -> None:
    if not isinstance(snapshot, dict):
        raise MalformedResponseError("snapshot must be a JSON object")
    state = snapshot.get("observation.state")
    action = snapshot.get("action")
    if not isinstance(state, list) or len(state) != state_dim:
        raise MalformedResponseError(f"snapshot observation.state must contain {state_dim} values")
    if not isinstance(action, list) or len(action) != action_dim:
        raise MalformedResponseError(f"snapshot action must contain {action_dim} values")
    for value in state + action:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise MalformedResponseError("snapshot values must be finite numbers")
    if "timestamp" not in snapshot or "sequence" not in snapshot:
        raise MalformedResponseError("snapshot missing timestamp or sequence")
    timestamp = snapshot["timestamp"]
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
    ):
        raise MalformedResponseError("snapshot timestamp must be a finite number")
    sequence = snapshot["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise MalformedResponseError("snapshot sequence must be a non-negative integer")


__all__ = [
    "ControlClient",
    "ControlClientError",
    "ControlServer",
    "DaemonDisconnectedError",
    "MalformedResponseError",
    "StaleSnapshotError",
]

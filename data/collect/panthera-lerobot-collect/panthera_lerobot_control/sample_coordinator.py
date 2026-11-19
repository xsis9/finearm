"""Pair one robot observation with the action from the same daemon snapshot."""

from __future__ import annotations

import copy
import threading


_lock = threading.Lock()
_snapshots: dict[str, dict] = {}


def publish_snapshot(socket_path: str, snapshot: dict) -> None:
    with _lock:
        _snapshots[str(socket_path)] = copy.deepcopy(snapshot)


def consume_snapshot(socket_path: str) -> dict | None:
    with _lock:
        snapshot = _snapshots.pop(str(socket_path), None)
    if snapshot is None:
        return None
    return copy.deepcopy(snapshot)


def clear_snapshot(socket_path: str) -> None:
    with _lock:
        _snapshots.pop(str(socket_path), None)


__all__ = ["clear_snapshot", "consume_snapshot", "publish_snapshot"]

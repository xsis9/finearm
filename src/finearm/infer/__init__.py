"""Inference domain: model adapters, chunk executors, and the ``finearm-infer`` CLI.

Adapters ("filename is the registry") declare how to talk to one model server;
executors drive a robot backend through them. Pure Python + numpy only at
import time; server connections happen in adapter constructors.
"""

from .adapters import AdapterBase, available as adapters_available, load as load_adapter, mock_obs
from .executors import ChunkExecutor, ExecutorStats, RTCExecutor, StopAndGoExecutor

__all__ = [
    "AdapterBase",
    "ChunkExecutor",
    "ExecutorStats",
    "RTCExecutor",
    "StopAndGoExecutor",
    "adapters_available",
    "load_adapter",
    "mock_obs",
]

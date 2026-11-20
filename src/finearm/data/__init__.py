"""Data-collection domain: collector plugins + the ``finearm-collect`` CLI."""

from .collector import CollectError, Collector, PolicySource, available, load

__all__ = ["CollectError", "Collector", "PolicySource", "available", "load"]

"""Zero-dependency core: config resolution, repo paths, plugin registry, norm stats.

``finearm.core`` must stay importable with a pure standard library — it is
installed into every environment (including the 4090 openpi venv), so a heavy
or broken import here would break everything (REFACTOR_PLAN.md design
principle 3). Module-level code must never read config files: resolution
happens inside functions only.
"""

from .config import load_jsonc, local_config_path, local_overrides, resolve, strip_json_comments
from .norm import QuantilePairs, load_quantile_pairs
from .paths import DEFAULT_PATHS, ROOT_ENV, default_path, find_repo_root, repo_root, resolve_path
from .registry import RegistryError, load_attr, scan_names

__all__ = [
    "DEFAULT_PATHS",
    "QuantilePairs",
    "RegistryError",
    "ROOT_ENV",
    "default_path",
    "find_repo_root",
    "load_attr",
    "load_jsonc",
    "load_quantile_pairs",
    "local_config_path",
    "local_overrides",
    "repo_root",
    "resolve",
    "resolve_path",
    "scan_names",
    "strip_json_comments",
]

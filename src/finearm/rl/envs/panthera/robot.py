"""Panthera-HT Stage-2 robot backend (execution machine, 4060).

Moved from ``RL/rlt_panthera/robot/panthera_robot.py`` in Phase 5a (that
path is now a shim). The machine-absolute acquisition-candidate list is
gone: the default acquisition backend is now a package module
(``finearm.infer.robot_interface``); PANTHERA_ACQ_IFACE and the
configs/local.jsonc ``acquisition_interface`` override still load a
custom file.

Wraps the acquisition machine's ``deploy/infer/robot_interface.py`` (uploaded
verbatim to /root/xsx/infer/robot_interface.py on the A100; NOT vendored —
loaded from the deploy tree at runtime so the acquisition implementation
stays the single source of truth) into the Stage-2 ``RobotInterface``
contract consumed by ``scripts/train_stage2.py``.

Adaptation decisions (2026-11-13):

* **raw frames**: the Stage-2 loop observes via ``get_obs(style="raw")``
  (640×480 originals). The acquisition class's ``"openpi"`` style pre-resizes
  to 224² with ``openpi_client.image_tools.resize_with_pad`` — a DIFFERENT
  kernel than the server-side JAX resize used by training/precompute, which
  would break the validated offline↔online distribution. The server resizes
  from raw frames exactly like the Phase-1 precompute did.
* **send_action has no read-back**: the arm daemon takes absolute targets
  and returns nothing. v1 records the REQUESTED target as the actual action
  (policy mode is assumed to execute commands faithfully); if the daemon
  clamps, replay actions carry that tiny error. Revisit if the daemon gains
  a read-back API.
* **HIL via daemon control modes** (2026-11-13, collection code uploaded):
  the panthera daemon owns BOTH arms and exposes ``teleop`` (leader→follower
  direct-follow, the exact mechanism the 64 episodes were collected with,
  including its own takeover smoothing — command-rate-limited from the
  follower's current pose) and ``freedrive`` (hand-teaching). The Stage-2
  loop's 'h' key switches the daemon to the HIL mode; the human action
  stream is the daemon snapshot's ``action`` field (7-dim absolute targets
  the daemon is commanding). Exit ('s'/'h') switches back to ``policy``:
  the daemon holds until our next policy action arrives (<0.25 s staleness),
  so the handoff is safe with no gap. ``poll_human`` stays (False, None) —
  automatic takeover detection would need the leader joints exposed in
  policy mode (a daemon change, deferred); entry is by key only.
* **reset**: reuses the acquisition ``_go_home`` interpolation (env
  PANTHERA_RTC_HOME, default "0,0,0,0,0,0,1.8").
* **estop**: best-effort ``client.hold()`` (stops policy following). The
  physical e-stop remains the real safety stop.

Env:
  PANTHERA_ACQ_IFACE   path to the acquisition robot_interface.py
                       (default: <repo>/infer/robot_interface.py via
                       finearm.core; see _acquisition_candidates)
  PANTHERA_MOCK_IMAGES "1" → zero-image mock cameras (link testing; the arm
                       daemon is still required)
  PANTHERA_RTC_HOME    home pose "j1,...,j6,gripper" (acquisition default)
  PANTHERA_HIL_MODE    "teleop" (default, leader arm) | "freedrive" (drag
                       teaching) — daemon mode switched into on HIL entry
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np

from finearm.rl.envs.robot_interface import RobotInterface as _Stage2Contract

logger = logging.getLogger(__name__)

# Four-layer path resolution for the acquisition-interface override
# (PANTHERA_ACQ_IFACE env > configs/local.jsonc).
from finearm.core import paths as _core_paths


def _load_acquisition_module():
    """PANTHERA_ACQ_IFACE / local.jsonc file override > finearm.infer package.

    The legacy machine-absolute candidate list (~/internship/...,
    /root/xsx/...) was removed in Phase 5a when the acquisition
    implementation moved into the package; a custom implementation is still
    injected per run by pointing either override at a file.
    """
    override = os.environ.get("PANTHERA_ACQ_IFACE")
    resolved = _core_paths.resolve_path("acquisition_interface", cli=override)
    if resolved is not None:
        path = Path(resolved).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"acquisition_interface override not found: {path}"
            )
        spec = importlib.util.spec_from_file_location("acq_robot_interface", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["acq_robot_interface"] = module
        spec.loader.exec_module(module)
        logger.info("Acquisition RobotInterface loaded from %s", path)
        return module
    from finearm.infer import robot_interface as module
    logger.info("Acquisition RobotInterface: finearm.infer.robot_interface")
    return module


class PantheraStage2Robot(_Stage2Contract):
    """Stage-2 contract adapter over the acquisition RobotInterface."""

    def __init__(self, *, home: str = "env", mock_images: bool | None = None):
        acq = _load_acquisition_module()
        self._acq = acq
        self._home_str = home
        if mock_images is None:
            mock_images = os.environ.get("PANTHERA_MOCK_IMAGES", "0") == "1"
        self._mock_images = mock_images
        self._iface = None
        self._home = None  # parsed 7-vector once connected

    # ── contract ────────────────────────────────────────────────────────
    def connect(self) -> None:
        # img_sizes=(): no client-side pre-resize; the server resizes raw
        # frames exactly like the training/precompute pipeline.
        self._iface = self._acq.RobotInterface(
            home=self._home_str, img_sizes=(), mock_images=self._mock_images,
        )
        home_str = os.environ.get("PANTHERA_RTC_HOME", self._acq.DEFAULT_HOME)
        if self._home_str not in ("env",):
            home_str = self._home_str
        if home_str:
            self._home = np.array([float(v) for v in home_str.split(",")], dtype=np.float64)
        logger.info("PantheraStage2Robot connected (mock_images=%s)", self._mock_images)

    def disconnect(self) -> None:
        if self._iface is not None:
            try:
                self._iface.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("acquisition close() failed: %s", exc)
            self._iface = None

    def get_observation(self) -> dict:
        raw = self._iface.get_obs("raw")
        return {
            "state": np.asarray(raw["state"], dtype=np.float32)[:7],
            "base_image": np.asarray(raw["front"]),    # 640×480 RGB uint8
            "wrist_image": np.asarray(raw["wrist"]),
        }

    def send_action(self, target: np.ndarray) -> np.ndarray:
        values = np.asarray(target, dtype=np.float64).ravel()[:7]
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite target: {values.tolist()}")
        self._iface.send_action(values)
        # No daemon read-back: record the requested target as actual (see
        # module docstring — revisit when the daemon returns the command).
        return values.astype(np.float32)

    # ── HIL: daemon control-mode switching (see module docstring) ───────
    def _hil_mode(self) -> str:
        mode = os.environ.get("PANTHERA_HIL_MODE", "teleop").strip().lower()
        if mode not in ("teleop", "freedrive"):
            logger.warning("unknown PANTHERA_HIL_MODE %r; falling back to teleop", mode)
            mode = "teleop"
        return mode

    def poll_human(self) -> tuple[bool, np.ndarray | None]:
        # No automatic detection in v1: leader joints are not exposed in the
        # daemon snapshot while in policy mode. Entry is via the 'h' key.
        return False, None

    def enter_human_control(self) -> bool:
        try:
            self._iface.client.set_mode(self._hil_mode())
            logger.info("HIL: daemon → %s (operator now controls the arm; "
                        "'s'/'h' to resume policy)", self._hil_mode())
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("HIL entry failed (mode switch to %s): %s", self._hil_mode(), exc)
            return False

    def exit_human_control(self) -> None:
        try:
            self._iface.client.set_mode("policy")
            logger.info("HIL exit: daemon → policy (holds until the next policy action)")
        except Exception as exc:  # noqa: BLE001
            logger.error("HIL exit mode switch failed: %s", exc)

    def latest_human_action(self) -> np.ndarray | None:
        try:
            snapshot = self._iface.client.snapshot(max_age_s=0.25)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HIL snapshot read failed: %s", exc)
            return None
        action = np.asarray(snapshot.get("action"), dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            return None
        return action

    def human_finished(self) -> bool:
        # Exit is manual only ('s'/'h'): the daemon snapshot does not expose
        # leader stillness in a way we can distinguish from holding.
        return False

    def reset(self) -> bool:
        if self._home is None:
            logger.info("reset skipped: no home pose configured")
            return False
        try:
            self._iface._go_home(self._home)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("reset failed: %s", exc)
            return False

    def estop(self) -> None:
        try:
            self._iface.client.hold()
            logger.warning("estop: daemon switched to HOLD (physical e-stop is the "
                           "authoritative stop)")
        except Exception as exc:  # noqa: BLE001
            logger.error("estop hold() failed: %s", exc)

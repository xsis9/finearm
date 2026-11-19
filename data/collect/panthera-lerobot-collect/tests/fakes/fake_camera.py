from __future__ import annotations

import time

from panthera_vla.fault_manager import FaultReason
from panthera_vla.observation_builder import CameraFrame


class FakeCameraManager:
    def __init__(self, reason=FaultReason.NONE, camera_ids=("front", "wrist")):
        self.reason = reason
        self.camera_ids = tuple(camera_ids)
        self.read_count = 0

    def read_latest(self):
        self.read_count += 1
        if self.reason is not FaultReason.NONE:
            return [], self.reason
        now = time.monotonic_ns()
        return [
            CameraFrame(camera_id, camera_id.encode("ascii"), "jpeg", 640, 480, now, self.read_count)
            for camera_id in self.camera_ids
        ], FaultReason.NONE

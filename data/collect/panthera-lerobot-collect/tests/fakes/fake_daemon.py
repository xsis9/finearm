from __future__ import annotations


class FakeDaemonClient:
    def __init__(self, state=None, action=None):
        self.actions = []
        self.hold_count = 0
        self.mode = "hold"
        self.snapshot_count = 0
        self.status_count = 0
        self.state = list(state if state is not None else [0.0] * 14)
        self.current_action = list(action if action is not None else self.state[:7])

    def status(self):
        self.status_count += 1
        return {"control_mode": self.mode, "holding": self.mode == "hold"}

    def snapshot(self, max_age_s=None):
        self.snapshot_count += 1
        return {
            "timestamp": 1.0,
            "sequence": self.snapshot_count,
            "observation.state": list(self.state),
            "action": list(self.current_action),
        }

    def hold(self):
        self.hold_count += 1
        self.mode = "hold"
        return {"control_mode": self.mode, "holding": True}

    def set_mode(self, mode):
        self.mode = mode
        return {"control_mode": self.mode, "holding": self.mode == "hold"}

    def send_policy_action(self, action):
        values = list(action)
        self.actions.append(values)
        self.current_action = values
        return {"control_mode": self.mode}

    def best_effort_hold_on_exit(self):
        self.hold()

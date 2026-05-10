"""Nominal omnidirectional SE(2) model for B2 high-level velocity control."""

from __future__ import annotations

import numpy as np


class OmniB2:
    """Integrates body-frame velocity commands into an SE(2) pose state."""

    state_dim = 6
    control_dim = 3

    def __init__(self, dt: float, max_vx: float, max_vy: float, max_wz: float) -> None:
        self.dt = float(dt)
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_wz = float(max_wz)
        self.state = np.zeros(self.state_dim, dtype=np.float32)

    def clip_control(self, u: np.ndarray) -> np.ndarray:
        control = np.asarray(u, dtype=np.float32).copy()
        control[0] = np.clip(control[0], -self.max_vx, self.max_vx)
        control[1] = np.clip(control[1], -self.max_vy, self.max_vy)
        control[2] = np.clip(control[2], -self.max_wz, self.max_wz)
        return control

    def update_state(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        next_state = np.asarray(state, dtype=np.float32).copy()
        vx, vy, wz = self.clip_control(control)
        theta = float(next_state[2])
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        next_state[0] += (vx * cos_theta - vy * sin_theta) * self.dt
        next_state[1] += (vx * sin_theta + vy * cos_theta) * self.dt
        next_state[2] += wz * self.dt
        next_state[3] = vx
        next_state[4] = vy
        next_state[5] = wz
        self.state = next_state
        return next_state

    def rollout(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=np.float32)
        states = np.zeros((controls.shape[0] + 1, self.state_dim), dtype=np.float32)
        states[0] = np.asarray(initial_state, dtype=np.float32)
        for idx, control in enumerate(controls):
            states[idx + 1] = self.update_state(states[idx], control)
        return states

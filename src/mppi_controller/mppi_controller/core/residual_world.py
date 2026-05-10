"""Oracle residual-world dynamics for omnidirectional SE(2) simulation."""

from __future__ import annotations

import numpy as np

from mppi_controller.core.omni_b2 import OmniB2
from mppi_controller.core.terrain import TerrainField


class ResidualWorld:
    """Apply terrain-dependent residuals to nominal velocity commands."""

    def __init__(
        self,
        model: OmniB2,
        terrain: TerrainField,
        *,
        enabled: bool = False,
        alpha: float = 0.35,
        residual_scale: float = 0.5,
        noise_std: float = 0.02,
        max_residual_ratio: float = 0.4,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.terrain = terrain
        self.enabled = bool(enabled)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.residual_scale = float(residual_scale)
        self.noise_std = float(noise_std)
        self.max_residual_ratio = float(max_residual_ratio)
        if self.residual_scale < 0.0:
            raise ValueError("residual_scale must be >= 0")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if self.max_residual_ratio < 0.0:
            raise ValueError("max_residual_ratio must be >= 0")
        self.rng = np.random.default_rng(seed)
        self.max_control = np.array(
            [model.max_vx, model.max_vy, model.max_wz],
            dtype=np.float32,
        )
        self.prev_u_real = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.prev_u_real[:] = 0.0

    @classmethod
    def from_config(
        cls,
        config: dict | None,
        *,
        model: OmniB2,
        terrain: TerrainField,
    ) -> "ResidualWorld":
        config = config or {}
        return cls(
            model,
            terrain,
            enabled=bool(config.get("enabled", False)),
            alpha=float(config.get("alpha", 0.35)),
            residual_scale=float(config.get("residual_scale", 0.5)),
            noise_std=float(config.get("noise_std", 0.02)),
            max_residual_ratio=float(config.get("max_residual_ratio", 0.4)),
            seed=int(config.get("seed", 0)),
        )

    def residual(self, state: np.ndarray, u_cmd: np.ndarray, terrain_features: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.zeros(3, dtype=np.float32)
        slope_f, slope_l, roughness, friction = (float(val) for val in terrain_features)
        u_cmd = np.asarray(u_cmd, dtype=np.float32)
        drag = -0.2 * (1.0 - friction) * u_cmd
        slope_bias = np.array(
            [
                -slope_f * (0.6 + 0.4 * abs(u_cmd[0])),
                -slope_l * (0.6 + 0.4 * abs(u_cmd[1])),
                -0.15 * roughness * np.sign(u_cmd[2]) * (0.3 + abs(u_cmd[2])),
            ],
            dtype=np.float32,
        )
        cross = np.array(
            [0.0, 0.05 * slope_l * u_cmd[0], 0.02 * slope_f * u_cmd[1]],
            dtype=np.float32,
        )
        base = slope_bias + drag + cross
        noise = self.rng.normal(
            loc=0.0,
            scale=self.noise_std * (1.0 + roughness),
            size=3,
        ).astype(np.float32)
        delta = self.residual_scale * base + noise * self.residual_scale
        max_delta = self.max_control * self.max_residual_ratio
        return np.clip(delta, -max_delta, max_delta)

    def update_state(
        self, state: np.ndarray, u_cmd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u_cmd = self.model.clip_control(u_cmd)
        terrain_features = self.terrain.feature(float(state[0]), float(state[1]))
        if not self.enabled:
            delta_u = np.zeros(3, dtype=np.float32)
            u_real = u_cmd
            next_state = self.model.update_state(state, u_real)
            return next_state, u_real, delta_u, terrain_features
        delta_u = self.residual(state, u_cmd, terrain_features)
        u_target = self.model.clip_control(u_cmd + delta_u)
        u_real = self.alpha * self.prev_u_real + (1.0 - self.alpha) * u_target
        u_real = self.model.clip_control(u_real)
        self.prev_u_real = u_real.copy()
        next_state = self.model.update_state(state, u_real)
        return next_state, u_real, delta_u, terrain_features

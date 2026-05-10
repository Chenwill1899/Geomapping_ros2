"""Terrain risk grid sampling utilities for sequence FDM."""

from __future__ import annotations

import numpy as np
import torch

from mppi_controller.core.terrain import TerrainField


def _vectorized_risk_cost(terrain: TerrainField, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Vectorized risk_cost over a meshgrid of points."""
    if not terrain.enabled:
        return np.zeros_like(grid_x, dtype=np.float32)

    # Base features (vectorized)
    slope_f = terrain.slope_scale * (np.sin(terrain.slope_wave * grid_x) + 0.35 * np.cos(0.5 * grid_y))
    slope_l = terrain.slope_scale * (np.cos(terrain.slope_wave * grid_y) + 0.35 * np.sin(0.5 * grid_x))
    roughness = terrain.roughness_scale * (0.5 + 0.5 * np.sin(terrain.roughness_wave * (grid_x + grid_y)))
    friction = terrain.friction_base - terrain.friction_slope_scale * (0.5 * (np.abs(slope_f) + np.abs(slope_l)))
    friction -= terrain.friction_roughness_scale * roughness

    # Noise (vectorized bilinear sample)
    if terrain._noise_grid is not None:
        noise, grad_x, grad_y = _batch_sample_noise(terrain, grid_x, grid_y)
        roughness += terrain.noise_roughness_weight * noise
        friction -= terrain.noise_friction_weight * noise
        slope_f += terrain.noise_slope_weight * grad_x
        slope_l += terrain.noise_slope_weight * grad_y

    roughness = np.clip(roughness, 0.0, 1.0)
    friction = np.clip(friction, 0.2, 1.0)

    # Goal relief
    attenuation = _batch_goal_relief(terrain, grid_x, grid_y)
    slope_f *= attenuation
    slope_l *= attenuation
    roughness *= attenuation
    friction = 1.0 - attenuation * (1.0 - friction)

    # Patches (vectorized per patch)
    for patch in terrain.patches:
        influence = _batch_patch_influence(patch, grid_x, grid_y)
        slope_f += influence * patch["slope_f_delta"]
        slope_l += influence * patch["slope_l_delta"]
        roughness += influence * patch["roughness_delta"]
        friction += influence * patch["friction_delta"]

    roughness = np.clip(roughness, 0.0, 1.0)
    friction = np.clip(friction, 0.2, 1.0)

    w0, w1, w2, w3 = terrain.risk_weights
    risk = w0 * np.abs(slope_f) + w1 * np.abs(slope_l) + w2 * roughness + w3 * (1.0 - friction)
    return risk.astype(np.float32)


def _batch_sample_noise(terrain: TerrainField, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized bilinear sample of noise grid."""
    x_min, x_max = terrain.noise_x_range
    y_min, y_max = terrain.noise_y_range
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    cols = terrain._noise_grid.shape[1]
    rows = terrain._noise_grid.shape[0]

    gx = np.clip((x - x_min) / x_span, 0.0, 1.0) * (cols - 1)
    gy = np.clip((y - y_min) / y_span, 0.0, 1.0) * (rows - 1)
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    x1 = np.minimum(x0 + 1, cols - 1)
    y1 = np.minimum(y0 + 1, rows - 1)
    tx = gx - x0
    ty = gy - y0

    def _sample_grid(grid):
        top = (1.0 - tx) * grid[y0, x0] + tx * grid[y0, x1]
        bottom = (1.0 - tx) * grid[y1, x0] + tx * grid[y1, x1]
        return (1.0 - ty) * top + ty * bottom

    return _sample_grid(terrain._noise_grid), _sample_grid(terrain._noise_grad_x), _sample_grid(terrain._noise_grad_y)


def _batch_goal_relief(terrain: TerrainField, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized goal relief attenuation."""
    relief = terrain.goal_relief
    if not bool(relief.get("enabled", False)):
        return np.ones_like(x, dtype=np.float32)
    center = relief.get("center", [0.0, 0.0])
    sigma = relief.get("sigma", [2.0, 1.2])
    sigma_x = max(1e-6, float(sigma[0]))
    sigma_y = max(1e-6, float(sigma[1]))
    strength = float(np.clip(relief.get("strength", 0.75), 0.0, 1.0))
    floor_val = float(np.clip(relief.get("floor", 0.25), 0.0, 1.0))
    dx = (x - float(center[0])) / sigma_x
    dy = (y - float(center[1])) / sigma_y
    gaussian = np.exp(-0.5 * (dx * dx + dy * dy))
    return np.maximum(floor_val, 1.0 - strength * gaussian).astype(np.float32)


def _batch_patch_influence(patch: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized patch influence."""
    patch_type = str(patch.get("type", "ellipse")).lower()
    angle = np.deg2rad(float(patch.get("angle", 0.0)))
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    dx = x - float(patch["center"][0])
    dy = y - float(patch["center"][1])
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    size = patch["size"]

    if patch_type == "ellipse":
        half_x = max(1e-6, 0.5 * float(size[0]))
        half_y = max(1e-6, 0.5 * float(size[1]))
        normalized_distance = np.sqrt((local_x / half_x) ** 2 + (local_y / half_y) ** 2)
        outside_distance = (normalized_distance - 1.0) * min(half_x, half_y)
    elif patch_type == "band":
        half_length = max(1e-6, 0.5 * float(size[0]))
        half_width = max(1e-6, 0.5 * float(size[1]))
        outside_distance = np.maximum(np.abs(local_x) - half_length, np.abs(local_y) - half_width)
    else:
        return np.zeros_like(x, dtype=np.float32)

    edge_width = float(patch.get("edge_width", 0.5))
    result = np.zeros_like(x, dtype=np.float32)
    mask_inside = outside_distance <= 0.0
    mask_outside = outside_distance >= edge_width
    mask_transition = ~mask_inside & ~mask_outside
    result[mask_inside] = 1.0
    t = np.clip(outside_distance[mask_transition] / edge_width, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    result[mask_transition] = 1.0 - smooth
    return result


def sample_terrain_risk_grid_np(
    terrain: TerrainField,
    x: float,
    y: float,
    size: int = 9,
    span: float = 18.0,
) -> np.ndarray:
    """Sample a flat risk grid centered on (x, y) using numpy (vectorized)."""
    half = span / 2.0
    xs = np.linspace(x - half, x + half, size, dtype=np.float32)
    ys = np.linspace(y - half, y + half, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    risk = _vectorized_risk_cost(terrain, grid_x, grid_y)
    return risk.reshape(-1)


def sample_terrain_risk_grid_torch(
    terrain: TerrainField,
    x: float,
    y: float,
    size: int = 9,
    span: float = 18.0,
) -> torch.Tensor:
    """Sample a flat risk grid centered on (x, y) using torch."""
    half = span / 2.0
    xs = torch.linspace(x - half, x + half, size, dtype=torch.float32)
    ys = torch.linspace(y - half, y + half, size, dtype=torch.float32)
    grid = torch.zeros((size, size), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = float(terrain.risk_cost(float(xs[i].item()), float(ys[j].item())))
    return grid.reshape(-1)

"""Random terrain generator for sequence FDM data collection."""

import numpy as np

from mppi_controller.core.terrain import TerrainField


class RandomTerrainGenerator:
    """Generates random TerrainField instances with varied patches."""

    def __init__(
        self,
        *,
        map_bounds: tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0),
        num_patches_range: tuple[int, int] = (2, 5),
        patch_size_range: tuple[float, float] = (2.0, 6.0),
        patch_types: tuple[str, ...] = ("ellipse", "band"),
        risk_intensity_range: tuple[float, float] = (0.3, 0.9),
        noise_enabled: bool = True,
        noise_scale_range: tuple[float, float] = (0.2, 0.5),
        noise_seed: int = 123,
    ) -> None:
        self.map_bounds = map_bounds
        self.num_patches_range = num_patches_range
        self.patch_size_range = patch_size_range
        self.patch_types = patch_types
        self.risk_intensity_range = risk_intensity_range
        self.noise_enabled = noise_enabled
        self.noise_scale_range = noise_scale_range
        self.noise_seed = noise_seed

    def generate(self, seed: int | None = None) -> TerrainField:
        rng = np.random.default_rng(seed)
        min_patches, max_patches = self.num_patches_range
        num_patches = rng.integers(min_patches, max_patches + 1)

        patches = []
        for i in range(num_patches):
            patch = self._generate_patch(rng, i)
            patches.append(patch)

        noise_scale = rng.uniform(*self.noise_scale_range)

        return TerrainField(
            enabled=True,
            patches=patches,
            # Moderate base terrain risk — patches still stand out
            slope_scale=0.06,
            slope_wave=0.6,
            roughness_scale=0.12,
            roughness_wave=0.45,
            friction_base=0.88,
            friction_slope_scale=0.10,
            friction_roughness_scale=0.06,
            noise_enabled=self.noise_enabled,
            noise_seed=seed if seed is not None else self.noise_seed,
            noise_scale=noise_scale,
            noise_x_range=(self.map_bounds[0], self.map_bounds[1]),
            noise_y_range=(self.map_bounds[2], self.map_bounds[3]),
        )

    def _generate_patch(self, rng: np.random.Generator, index: int) -> dict:
        x_min, x_max, y_min, y_max = self.map_bounds
        center_x = rng.uniform(x_min, x_max)
        center_y = rng.uniform(y_min, y_max)
        size_x = rng.uniform(*self.patch_size_range)
        size_y = rng.uniform(*self.patch_size_range)
        angle = rng.uniform(0.0, 360.0)
        edge_width = 0.5
        patch_type = rng.choice(self.patch_types)
        intensity = rng.uniform(*self.risk_intensity_range)

        return {
            "name": f"patch_{index}_{patch_type}",
            "type": patch_type,
            "center": [float(center_x), float(center_y)],
            "angle": float(angle),
            "size": [float(size_x), float(size_y)],
            "edge_width": float(edge_width),
            "slope_f_delta": float(intensity * 0.8),
            "slope_l_delta": float(intensity * 0.6),
            "roughness_delta": float(intensity * 0.5),
            "friction_delta": float(-intensity * 0.4),
        }

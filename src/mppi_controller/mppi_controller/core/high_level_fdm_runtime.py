"""TorchScript runtime for high_level_fdm exports.

This module intentionally does not import high_level_fdm. The ROS-side controller
only depends on a TorchScript file and its JSON metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class HighLevelFdmMetadata:
    horizon: int
    history_len: int
    map_size: int
    map_channels: int
    dt: float
    risk_names: tuple[str, ...]
    map_max_cost: float = 100.0

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "HighLevelFdmMetadata":
        required = ("horizon", "history_len", "map_size", "map_channels", "dt", "risk_names")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"fdm_metadata.json missing required keys: {missing}")
        risk_names = tuple(str(name) for name in raw["risk_names"])
        if not risk_names:
            raise ValueError("fdm_metadata.json risk_names must be non-empty")
        return HighLevelFdmMetadata(
            horizon=int(raw["horizon"]),
            history_len=int(raw["history_len"]),
            map_size=int(raw["map_size"]),
            map_channels=int(raw["map_channels"]),
            dt=float(raw["dt"]),
            risk_names=risk_names,
            map_max_cost=float(raw.get("map_max_cost", 100.0)),
        )


class HighLevelFdmRuntime:
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        metadata: HighLevelFdmMetadata,
        *,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.metadata = metadata
        self.risk_any_index = (
            metadata.risk_names.index("any")
            if "any" in metadata.risk_names
            else len(metadata.risk_names) - 1
        )

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        model_file: str = "fdm_ts.pt",
        metadata_file: str = "fdm_metadata.json",
    ) -> "HighLevelFdmRuntime":
        model_dir = Path(model_dir)
        metadata_path = model_dir / metadata_file
        model_path = model_dir / model_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing high_level_fdm metadata: {metadata_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"missing high_level_fdm TorchScript model: {model_path}")
        metadata = HighLevelFdmMetadata.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
        model = torch.jit.load(str(model_path), map_location=device)
        return cls(model, metadata, device=device)

    @torch.inference_mode()
    def predict(
        self,
        history: torch.Tensor,
        local_map: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = history.to(device=self.device, dtype=torch.float32, non_blocking=True)
        local_map = local_map.to(device=self.device, dtype=torch.float32, non_blocking=True)
        actions = actions.to(device=self.device, dtype=torch.float32, non_blocking=True)
        pose, risk, applied_twist = self.model(history, local_map, actions)
        return pose.to(torch.float32), risk.to(torch.float32), applied_twist.to(torch.float32)

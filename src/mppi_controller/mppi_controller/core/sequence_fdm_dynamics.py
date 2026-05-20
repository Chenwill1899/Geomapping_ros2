"""SequenceFdmDynamics: inference-time wrapper for SequenceFdmMlpV2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mppi_controller.core.sequence_fdm_v2 import COSTMAP_GRID_DIM, SequenceFdmMlpV2


class SequenceFdmDynamics:
    """Loads a trained SequenceFdmMlpV2 and provides normalized inference."""

    def __init__(
        self,
        model: SequenceFdmMlpV2,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        control_mean: np.ndarray,
        control_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.horizon_steps = model.horizon_steps

        self.state_mean = torch.from_numpy(state_mean).to(device)
        self.state_std = torch.from_numpy(state_std).to(device)
        self.control_mean = torch.from_numpy(control_mean).to(device)
        self.control_std = torch.from_numpy(control_std).to(device)
        target_mean = torch.from_numpy(target_mean).to(device)
        target_std = torch.from_numpy(target_std).to(device)
        # target = [state_x, state_y, ..., state_wz for each step, then risk logits]
        state_target_len = self.horizon_steps * 6
        self.state_target_mean = target_mean[:state_target_len].view(self.horizon_steps, 6)
        self.state_target_std = target_std[:state_target_len].view(self.horizon_steps, 6)

    @classmethod
    def from_artifacts(cls, model_dir: Path, device: str = "cpu") -> "SequenceFdmDynamics":
        model_dir = Path(model_dir)
        ckpt_path = model_dir / "best_model.pt"
        norm_path = model_dir / "normalization.npz"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization not found: {norm_path}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        norm = np.load(norm_path)

        horizon_steps = checkpoint["horizon_steps"]
        hidden_dims = checkpoint.get("hidden_dims", [256, 256, 256])
        input_dim = checkpoint.get("input_dim", 6 + 3 * horizon_steps + COSTMAP_GRID_DIM)
        target_dim = checkpoint.get("target_dim", horizon_steps * 7)

        expected_input_dim = 6 + 3 * horizon_steps + COSTMAP_GRID_DIM
        expected_target_dim = horizon_steps * 7
        if input_dim != expected_input_dim:
            raise ValueError(
                f"Checkpoint input_dim {input_dim} does not match expected {expected_input_dim}"
            )
        if target_dim != expected_target_dim:
            raise ValueError(
                f"Checkpoint target_dim {target_dim} does not match expected {expected_target_dim}"
            )

        model = SequenceFdmMlpV2(horizon_steps=horizon_steps, hidden_dims=hidden_dims)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(
            model=model,
            state_mean=norm["state_mean"].astype(np.float32),
            state_std=norm["state_std"].astype(np.float32),
            control_mean=norm["control_mean"].astype(np.float32),
            control_std=norm["control_std"].astype(np.float32),
            target_mean=norm["target_mean"].astype(np.float32),
            target_std=norm["target_std"].astype(np.float32),
            device=device,
        )

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def _normalize_controls(self, controls: torch.Tensor) -> torch.Tensor:
        return (controls - self.control_mean) / self.control_std

    def _denormalize_state_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets * self.state_target_std + self.state_target_mean

    def predict_torch(
        self,
        state: torch.Tensor,
        controls: torch.Tensor,
        costmap_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch inference with torch tensors.

        Args:
            state: (B, 6) or (6,)
            controls: (B, H, 3) or (H, 3)
            costmap_grid: (B, 81) or (81,)

        Returns:
            states_pred: (B, H, 6) or (H, 6)
            risk_logits: (B, H) or (H,)
        """
        squeeze = state.ndim == 1
        if squeeze:
            state = state.unsqueeze(0)
            controls = controls.unsqueeze(0)
            costmap_grid = costmap_grid.unsqueeze(0)

        state = state.to(self.device)
        controls = controls.to(self.device)
        costmap_grid = costmap_grid.to(self.device)

        state_norm = self._normalize_state(state)
        controls_norm = self._normalize_controls(controls)

        with torch.no_grad():
            out = self.model(state_norm, controls_norm, costmap_grid)

        states_pred_raw, risk_logits = out
        states_pred = self._denormalize_state_targets(states_pred_raw)

        if squeeze:
            states_pred = states_pred.squeeze(0)
            risk_logits = risk_logits.squeeze(0)

        return states_pred, risk_logits

    def predict(
        self,
        state: np.ndarray,
        controls: np.ndarray,
        costmap_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-sample inference with numpy arrays.

        Args:
            state: (6,)
            controls: (H, 3)
            costmap_grid: (81,)

        Returns:
            states_pred: (H, 6)
            risk_logits: (H,)
        """
        state_t = torch.from_numpy(state).float()
        controls_t = torch.from_numpy(controls).float()
        costmap_t = torch.from_numpy(costmap_grid).float()

        states_pred, risk_logits = self.predict_torch(state_t, controls_t, costmap_t)
        return states_pred.cpu().numpy(), risk_logits.cpu().numpy()

    def predict_batch(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        costmaps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch inference with numpy arrays.

        Args:
            states: (B, 6)
            controls: (B, H, 3)
            costmaps: (B, 81)

        Returns:
            states_pred: (B, H, 6)
            risk_logits: (B, H)
        """
        states_t = torch.from_numpy(states).float()
        controls_t = torch.from_numpy(controls).float()
        costmaps_t = torch.from_numpy(costmaps).float()

        states_pred, risk_logits = self.predict_torch(states_t, controls_t, costmaps_t)
        return states_pred.cpu().numpy(), risk_logits.cpu().numpy()

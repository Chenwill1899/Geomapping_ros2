"""Learned residual dynamics wrapper for Stage 5 NumPy MPPI rollout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mppi_controller.core.omni_b2 import OmniB2
from mppi_controller.core.residual_fdm_model import (
    FEATURE_NAMES,
    TARGET_NAMES,
    SequenceFdmMlp,
    sequence_fdm_feature_names,
    sequence_fdm_target_names,
    ResidualFdmMlp,
    build_feature_vector,
)
from mppi_controller.core.terrain import TerrainField


@dataclass(frozen=True)
class LearnedResidualStep:
    next_state: np.ndarray
    real_control: np.ndarray
    predicted_residual: np.ndarray
    terrain_features: np.ndarray
    terrain_risk: float


@dataclass(frozen=True)
class LearnedSequencePrediction:
    rel_trajectory: np.ndarray
    terrain_risk: np.ndarray


class LearnedResidualDynamics:
    """Loads a residual FDM checkpoint and applies it as velocity correction."""

    def __init__(
        self,
        *,
        model: ResidualFdmMlp,
        robot: OmniB2,
        terrain: TerrainField,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        device: torch.device,
        checkpoint_path: Path,
        normalization_path: Path,
    ) -> None:
        self.model = model
        self.robot = robot
        self.terrain = terrain
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.normalization_path = Path(normalization_path)

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path,
        *,
        robot: OmniB2,
        terrain: TerrainField,
        checkpoint: str | Path = "best_model.pt",
        normalization: str | Path = "normalization.npz",
        device: str = "cpu",
    ) -> "LearnedResidualDynamics":
        model_dir = Path(model_dir)
        checkpoint_path = _resolve_artifact_path(model_dir, checkpoint)
        normalization_path = _resolve_artifact_path(model_dir, normalization)
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        input_dim = int(checkpoint_data["input_dim"])
        hidden_dim = int(checkpoint_data["hidden_dim"])
        if input_dim != len(FEATURE_NAMES):
            raise ValueError(f"Expected input_dim={len(FEATURE_NAMES)}, checkpoint has input_dim={input_dim}")
        _validate_names(checkpoint_data.get("feature_names"), FEATURE_NAMES, "feature_names")
        _validate_names(checkpoint_data.get("target_names"), TARGET_NAMES, "target_names")

        normalizer = np.load(normalization_path)
        feature_mean = np.asarray(normalizer["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(normalizer["feature_std"], dtype=np.float32)
        target_mean = np.asarray(normalizer["target_mean"], dtype=np.float32)
        target_std = np.asarray(normalizer["target_std"], dtype=np.float32)
        if feature_mean.shape != (len(FEATURE_NAMES),) or feature_std.shape != (len(FEATURE_NAMES),):
            raise ValueError("normalization feature_mean/feature_std shape does not match residual FDM input schema")
        if target_mean.shape != (len(TARGET_NAMES),) or target_std.shape != (len(TARGET_NAMES),):
            raise ValueError("normalization target_mean/target_std shape does not match residual FDM target schema")
        if "feature_names" in normalizer:
            _validate_names(normalizer["feature_names"], FEATURE_NAMES, "normalization feature_names")
        if "target_names" in normalizer:
            _validate_names(normalizer["target_names"], TARGET_NAMES, "normalization target_names")

        torch_device = torch.device(device)
        model = ResidualFdmMlp(input_dim=input_dim, hidden_dim=hidden_dim).to(torch_device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            robot=robot,
            terrain=terrain,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            device=torch_device,
            checkpoint_path=checkpoint_path,
            normalization_path=normalization_path,
        )

    def predict_residual(
        self,
        state: np.ndarray,
        command: np.ndarray,
        terrain_features: np.ndarray | None = None,
        terrain_risk: float | None = None,
    ) -> np.ndarray:
        residuals = self.predict_residual_batch(
            np.asarray(state, dtype=np.float32).reshape(1, 6),
            np.asarray(command, dtype=np.float32).reshape(1, 3),
            None if terrain_features is None else np.asarray(terrain_features, dtype=np.float32).reshape(1, 4),
            None if terrain_risk is None else np.asarray([terrain_risk], dtype=np.float32),
        )
        return residuals[0]

    def predict_residual_batch(
        self,
        states: np.ndarray,
        commands: np.ndarray,
        terrain_features: np.ndarray | None = None,
        terrain_risk: np.ndarray | None = None,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32).reshape(-1, 6)
        commands = np.asarray(commands, dtype=np.float32).reshape(-1, 3)
        if len(states) != len(commands):
            raise ValueError("states and commands must have the same batch length")
        features, risks = self._terrain_batch(states, terrain_features, terrain_risk)
        fdm_features = np.concatenate([states, commands, features, risks[:, None]], axis=1).astype(
            np.float32,
            copy=False,
        )
        standardized = ((fdm_features - self.feature_mean) / self.feature_std).astype(np.float32, copy=False)
        tensor = torch.as_tensor(standardized, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(tensor).detach().cpu().numpy()
        return (pred * self.target_std + self.target_mean).astype(np.float32, copy=False)

    def step(self, state: np.ndarray, command: np.ndarray) -> LearnedResidualStep:
        state = np.asarray(state, dtype=np.float32).reshape(6)
        command = self.robot.clip_control(command)
        terrain_features = self.terrain.feature(float(state[0]), float(state[1]))
        terrain_risk = self.terrain.risk_cost(float(state[0]), float(state[1]), features=terrain_features)
        residual = self.predict_residual(state, command, terrain_features, terrain_risk)
        real_control = self.robot.clip_control(command + residual)
        next_state = self.robot.update_state(state, real_control)
        return LearnedResidualStep(
            next_state=next_state,
            real_control=real_control,
            predicted_residual=residual,
            terrain_features=terrain_features,
            terrain_risk=float(terrain_risk),
        )

    def _terrain_batch(
        self,
        states: np.ndarray,
        terrain_features: np.ndarray | None,
        terrain_risk: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if terrain_features is None:
            features = np.asarray(
                [self.terrain.feature(float(state[0]), float(state[1])) for state in states],
                dtype=np.float32,
            )
        else:
            features = np.asarray(terrain_features, dtype=np.float32).reshape(len(states), 4)
        if terrain_risk is None:
            risks = np.asarray(
                [
                    self.terrain.risk_cost(float(state[0]), float(state[1]), features=features[idx])
                    for idx, state in enumerate(states)
                ],
                dtype=np.float32,
            )
        else:
            risks = np.asarray(terrain_risk, dtype=np.float32).reshape(len(states))
        return features, risks


class LearnedSequenceResidualDynamics:
    """Loads a sequence FDM checkpoint and predicts a horizon of residual-relative motion."""

    def __init__(
        self,
        *,
        model: SequenceFdmMlp,
        robot: OmniB2,
        terrain: TerrainField,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        sequence_horizon: int,
        include_history_controls: bool,
        history_steps: int,
        device: torch.device,
        checkpoint_path: Path,
        normalization_path: Path,
    ) -> None:
        self.model = model
        self.robot = robot
        self.terrain = terrain
        self.sequence_horizon = int(sequence_horizon)
        self.include_history_controls = bool(include_history_controls)
        self.history_steps = int(history_steps)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.normalization_path = Path(normalization_path)

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path,
        *,
        robot: OmniB2,
        terrain: TerrainField,
        checkpoint: str | Path = "best_model.pt",
        normalization: str | Path = "normalization.npz",
        device: str = "cpu",
        sequence_horizon: int | None = None,
        include_history_controls: bool = True,
        history_steps: int = 1,
    ) -> "LearnedSequenceResidualDynamics":
        model_dir = Path(model_dir)
        checkpoint_path = _resolve_artifact_path(model_dir, checkpoint)
        normalization_path = _resolve_artifact_path(model_dir, normalization)
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        input_dim = int(checkpoint_data["input_dim"])
        hidden_dim = int(checkpoint_data["hidden_dim"])
        if "target_dim" in checkpoint_data:
            target_dim = int(checkpoint_data["target_dim"])
        else:
            target_names_payload = checkpoint_data.get("target_names")
            target_dim = int(len(target_names_payload)) if target_names_payload is not None else 0
        if target_dim is None or not np.isfinite(target_dim):
            target_names_payload = checkpoint_data.get("target_names")
            target_dim = int(len(target_names_payload)) if target_names_payload is not None else 0
        if not target_dim:
            raise ValueError("sequence_fdm checkpoint must define target_dim")
        if target_dim % 4 != 0:
            raise ValueError("sequence_fdm checkpoint target_dim must be a multiple of 4")
        resolved_horizon = int(checkpoint_data.get("sequence_horizon", sequence_horizon or (target_dim // 4)))
        if resolved_horizon <= 0:
            resolved_horizon = target_dim // 4
        if sequence_horizon is not None and int(sequence_horizon) != resolved_horizon:
            raise ValueError(
                f"configured sequence_horizon={sequence_horizon} does not match checkpoint sequence_horizon={resolved_horizon}"
            )
        if include_history_controls and history_steps <= 0:
            raise ValueError("history_steps must be > 0 when include_history_controls is true")

        feature_names = sequence_fdm_feature_names(
            resolved_horizon,
            include_history_controls=include_history_controls,
            include_history_steps=history_steps,
        )
        target_names = sequence_fdm_target_names(resolved_horizon)
        if input_dim != len(feature_names):
            raise ValueError(
                f"Expected input_dim={len(feature_names)} for sequence_fdm, checkpoint has input_dim={input_dim}"
            )
        if target_dim != len(target_names):
            raise ValueError(
                f"Expected target_dim={len(target_names)} for sequence_fdm, checkpoint has target_dim={target_dim}"
            )
        _validate_names(checkpoint_data.get("feature_names"), feature_names, "feature_names")
        _validate_names(checkpoint_data.get("target_names"), target_names, "target_names")

        normalizer = np.load(normalization_path)
        feature_mean = np.asarray(normalizer["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(normalizer["feature_std"], dtype=np.float32)
        target_mean = np.asarray(normalizer["target_mean"], dtype=np.float32)
        target_std = np.asarray(normalizer["target_std"], dtype=np.float32)
        if feature_mean.shape != (len(feature_names),) or feature_std.shape != (len(feature_names),):
            raise ValueError("normalization feature_mean/feature_std shape does not match sequence FDM input schema")
        if target_mean.shape != (len(target_names),) or target_std.shape != (len(target_names),):
            raise ValueError("normalization target_mean/target_std shape does not match sequence FDM target schema")
        if "feature_names" in normalizer:
            _validate_names(normalizer["feature_names"], feature_names, "normalization feature_names")
        if "target_names" in normalizer:
            _validate_names(normalizer["target_names"], target_names, "normalization target_names")

        torch_device = torch.device(device)
        model = SequenceFdmMlp(
            input_dim=input_dim,
            output_horizon=resolved_horizon,
            hidden_dim=hidden_dim,
        ).to(torch_device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            robot=robot,
            terrain=terrain,
            sequence_horizon=resolved_horizon,
            include_history_controls=include_history_controls,
            history_steps=history_steps,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            device=torch_device,
            checkpoint_path=checkpoint_path,
            normalization_path=normalization_path,
        )

    def predict_sequence(
        self,
        state: np.ndarray,
        command_sequence: np.ndarray,
        history: np.ndarray | None = None,
    ) -> LearnedSequencePrediction:
        rel, risk = self.predict_sequence_batch(
            np.asarray(state, dtype=np.float32).reshape(1, 6),
            np.asarray(command_sequence, dtype=np.float32).reshape(1, self.sequence_horizon, 3),
            None if history is None else np.asarray(history, dtype=np.float32).reshape(1, -1),
        )
        return LearnedSequencePrediction(rel_trajectory=rel[0], terrain_risk=risk[0])

    def predict_sequence_batch(
        self,
        states: np.ndarray,
        command_sequences: np.ndarray,
        history: np.ndarray | None = None,
        terrain_features: np.ndarray | None = None,
        terrain_risk: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32).reshape(-1, 6)
        command_sequences = np.asarray(command_sequences, dtype=np.float32).reshape(-1, self.sequence_horizon, 3)
        if command_sequences.shape[1] != self.sequence_horizon:
            raise ValueError(
                f"command_sequences horizon mismatch: got {command_sequences.shape[1]}, expected {self.sequence_horizon}"
            )
        if len(states) != command_sequences.shape[0]:
            raise ValueError("states and command_sequences must have same batch length")
        history = self._prepare_history(states, history)
        features, risks = self._terrain_batch(states, terrain_features, terrain_risk)
        terrain_step = np.concatenate([features, risks[:, None]], axis=1)
        terrain_seq = np.repeat(terrain_step[:, None, :], self.sequence_horizon, axis=1)
        per_step = np.concatenate([command_sequences, terrain_seq], axis=2)
        feature_block = np.concatenate(
            [
                states,
                history,
                per_step.reshape(len(states), -1),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        if feature_block.shape[1] != self.feature_mean.shape[0]:
            raise ValueError("sequence feature block does not match model schema")
        standardized = ((feature_block - self.feature_mean) / self.feature_std).astype(np.float32, copy=False)
        tensor = torch.as_tensor(standardized, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(tensor).detach().cpu().numpy()
        pred = pred * self.target_std + self.target_mean
        pred = pred.reshape(len(states), self.sequence_horizon, 4).astype(np.float32, copy=False)
        return pred[:, :, :3], pred[:, :, 3]

    def predict_sequence_batch_torch(
        self,
        states: torch.Tensor,
        command_sequences: torch.Tensor,
        history: torch.Tensor | None = None,
        terrain_features: torch.Tensor | None = None,
        terrain_risk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states_t = states.to(torch.float32).reshape(-1, 6)
        command_sequences_t = command_sequences.to(torch.float32).reshape(-1, self.sequence_horizon, 3)
        if command_sequences_t.shape[1] != self.sequence_horizon:
            raise ValueError(
                f"command_sequences horizon mismatch: got {command_sequences_t.shape[1]}, expected {self.sequence_horizon}"
            )
        if states_t.shape[0] != command_sequences_t.shape[0]:
            raise ValueError("states and command_sequences must have same batch length")
        history_t = self._prepare_history_torch(states_t, history)
        features, risks = self._terrain_batch_torch(states_t, terrain_features, terrain_risk)
        terrain_step = torch.cat([features, risks.unsqueeze(1)], dim=1)
        terrain_seq = terrain_step.unsqueeze(1).repeat(1, self.sequence_horizon, 1)
        per_step = torch.cat([command_sequences_t, terrain_seq], dim=2)
        feature_block = torch.cat(
            [
                states_t,
                history_t,
                per_step.reshape(len(states_t), -1),
            ],
            dim=1,
        ).to(torch.float32)
        if feature_block.shape[1] != self.feature_mean.shape[0]:
            raise ValueError("sequence feature block does not match model schema")
        mean_t = self._to_tensor(self.feature_mean)
        std_t = self._to_tensor(self.feature_std)
        standardized = (feature_block - mean_t) / std_t
        self.model.eval()
        with torch.no_grad():
            pred = self.model(standardized)
        pred = pred * self._to_tensor(self.target_std) + self._to_tensor(self.target_mean)
        pred = pred.reshape(len(states_t), self.sequence_horizon, 4)
        return pred[:, :, :3].to(torch.float32), pred[:, :, 3].to(torch.float32)

    def _prepare_history(self, states: np.ndarray, history: np.ndarray | None) -> np.ndarray:
        num_samples = len(states)
        if not self.include_history_controls:
            return np.zeros((num_samples, 0), dtype=np.float32)
        if history is None:
            return np.zeros((num_samples, self.history_steps * 3), dtype=np.float32)
        history_arr = np.asarray(history, dtype=np.float32)
        if history_arr.ndim == 1:
            history_arr = history_arr.reshape(1, -1)
        else:
            history_arr = history_arr.reshape(history_arr.shape[0], -1)
        if history_arr.size == 0:
            return np.zeros((num_samples, self.history_steps * 3), dtype=np.float32)
        if history_arr.shape[1] == 3:
            history_arr = np.repeat(history_arr, self.history_steps, axis=1)
        elif history_arr.shape[1] != self.history_steps * 3:
            raise ValueError(
                f"history controls must be shape [n,3] or [n,{self.history_steps * 3}], got {history_arr.shape}"
            )
        if history_arr.shape[0] == 1 and num_samples > 1:
            history_arr = np.repeat(history_arr, num_samples, axis=0)
        if history_arr.shape[0] != num_samples:
            raise ValueError(
                f"history controls batch size {history_arr.shape[0]} does not match batch size {num_samples}"
            )
        return history_arr[:, : self.history_steps * 3]

    def _prepare_history_torch(
        self,
        states: torch.Tensor,
        history: torch.Tensor | None,
    ) -> torch.Tensor:
        num_samples = int(states.shape[0])
        if not self.include_history_controls:
            return torch.zeros((num_samples, 0), dtype=torch.float32, device=self.device)
        if history is None:
            return torch.zeros((num_samples, self.history_steps * 3), dtype=torch.float32, device=self.device)
        history_t = history.to(torch.float32)
        if history_t.ndim == 1:
            history_t = history_t.reshape(1, -1)
        else:
            history_t = history_t.reshape(history_t.shape[0], -1)
        if history_t.numel() == 0:
            return torch.zeros((num_samples, self.history_steps * 3), dtype=torch.float32, device=self.device)
        if history_t.shape[1] == 3:
            history_t = history_t.repeat(1, self.history_steps)
        elif history_t.shape[1] != self.history_steps * 3:
            raise ValueError(
                f"history controls must be shape [n,3] or [n,{self.history_steps * 3}], got {tuple(history_t.shape)}"
            )
        if history_t.shape[0] == 1 and num_samples > 1:
            history_t = history_t.expand(num_samples, -1)
        if history_t.shape[0] != num_samples:
            raise ValueError(
                f"history controls batch size {history_t.shape[0]} does not match batch size {num_samples}"
            )
        return history_t[:, : self.history_steps * 3].to(self.device, dtype=torch.float32)

    def _terrain_batch_torch(
        self,
        states: torch.Tensor,
        terrain_features: torch.Tensor | None,
        terrain_risk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if terrain_features is None:
            features_np = np.asarray([self.terrain.feature(float(s[0]), float(s[1])) for s in states.detach().cpu().numpy()])
            features = torch.as_tensor(features_np, dtype=torch.float32, device=self.device)
        else:
            features = terrain_features.to(torch.float32).to(self.device)
            features = features.reshape(len(states), 4)
        if terrain_risk is None:
            risks_np = np.asarray(
                [
                    self.terrain.risk_cost(float(state[0]), float(state[1]), features=features[idx].detach().cpu().numpy())
                    for idx, state in enumerate(states.detach().cpu().numpy())
                ],
                dtype=np.float32,
            )
            risks = torch.as_tensor(risks_np, dtype=torch.float32, device=self.device)
        else:
            risks = terrain_risk.to(torch.float32).to(self.device).reshape(len(states))
        return features, risks

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)


def _resolve_artifact_path(model_dir: Path, artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return model_dir / path


def _validate_names(actual, expected: list[str], label: str) -> None:
    if actual is None:
        return
    actual_list = [str(item) for item in np.asarray(actual).tolist()]
    if actual_list != list(expected):
        raise ValueError(f"{label} do not match residual FDM schema")

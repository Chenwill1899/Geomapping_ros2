"""Training utilities for Sequence FDM V2 with costmap-grid input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mppi_controller.core.sequence_fdm_v2 import COSTMAP_GRID_DIM, GOAL_PATH_FEATURE_DIM


class SequenceFdmDataset(Dataset):
    """PyTorch dataset for sequence FDM V2 training windows."""

    def __init__(self, windows: list[dict], horizon_steps: int) -> None:
        self.windows = windows
        self.horizon_steps = horizon_steps

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        w = self.windows[idx]
        state = torch.as_tensor(w["state"], dtype=torch.float32)
        controls = torch.as_tensor(w["controls"], dtype=torch.float32)
        costmap_grid = torch.as_tensor(_window_costmap_grid(w), dtype=torch.float32)
        goal_path_features = torch.as_tensor(_window_goal_path_features(w), dtype=torch.float32)
        target_states = torch.as_tensor(w["target_states"], dtype=torch.float32)
        target_risk = torch.as_tensor(w["target_risk"], dtype=torch.float32)
        return state, controls, costmap_grid, goal_path_features, target_states, target_risk


def _window_costmap_grid(window: dict) -> np.ndarray:
    if "costmap_grid" in window:
        grid = np.asarray(window["costmap_grid"], dtype=np.float32)
    else:
        grid = np.asarray(window["terrain_grid"], dtype=np.float32)
    if grid.reshape(-1).shape != (COSTMAP_GRID_DIM,):
        raise ValueError(f"costmap_grid must be flat {COSTMAP_GRID_DIM}-D, got shape {grid.shape}")
    return grid.reshape(COSTMAP_GRID_DIM)


def _window_goal_path_features(window: dict) -> np.ndarray:
    features = np.asarray(window.get("goal_path_features", np.zeros(GOAL_PATH_FEATURE_DIM)), dtype=np.float32)
    if features.reshape(-1).shape != (GOAL_PATH_FEATURE_DIM,):
        raise ValueError(f"goal_path_features must be flat {GOAL_PATH_FEATURE_DIM}-D, got shape {features.shape}")
    return features.reshape(GOAL_PATH_FEATURE_DIM)


def compute_normalization(windows: list[dict], horizon_steps: int) -> dict[str, np.ndarray]:
    """Compute mean/std for states, controls, and targets from training windows."""
    states = np.stack([w["state"] for w in windows])
    controls = np.stack([w["controls"] for w in windows])
    target_states = np.stack([w["target_states"].reshape(-1) for w in windows])
    target_risk = np.stack([w["target_risk"] for w in windows])

    target = np.concatenate([target_states, target_risk], axis=1)

    return {
        "state_mean": np.mean(states, axis=0).astype(np.float32),
        "state_std": np.std(states, axis=0).astype(np.float32) + 1e-8,
        "control_mean": np.mean(controls, axis=(0, 1)).astype(np.float32),
        "control_std": np.std(controls, axis=(0, 1)).astype(np.float32) + 1e-8,
        "target_mean": np.mean(target, axis=0).astype(np.float32),
        "target_std": np.std(target, axis=0).astype(np.float32) + 1e-8,
        "horizon_steps": np.array([horizon_steps], dtype=np.int32),
    }


def train_sequence_fdm_v2(
    windows: list[dict],
    output_dir: str | Path,
    hidden_dims: list[int] | None = None,
    curriculum_phases: list[tuple[int, int, float]] | None = None,
    batch_size: int = 64,
    w_traj: float = 1.0,
    w_risk: float = 0.5,
    val_ratio: float = 0.15,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_tensorboard: bool = True,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train SequenceFdmMlpV2 with curriculum learning.

    Args:
        windows: List of training windows from build_sequence_fdm_windows
        output_dir: Directory to save checkpoints and normalization
        hidden_dims: MLP hidden layer sizes
        curriculum_phases: List of (horizon_steps, epochs, lr) tuples
        batch_size: Training batch size
        w_traj: Trajectory loss weight
        w_risk: Risk loss weight
        val_ratio: Fraction of windows for validation
        patience: Early stopping patience (epochs)
        device: "cuda" or "cpu"
        use_tensorboard: Whether to log metrics to TensorBoard
        resume_from: Checkpoint directory to resume from (loads best_model.pt + optimizer.pt)
    """
    from mppi_controller.core.sequence_fdm_v2 import SequenceFdmMlpV2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hidden_dims is None:
        hidden_dims = [256, 256, 256]
    if curriculum_phases is None:
        curriculum_phases = [(5, 50, 1e-3), (10, 50, 5e-4), (20, 100, 1e-4), (20, 50, 5e-5)]

    # Split train/val
    n_val = int(len(windows) * val_ratio)
    rng = np.random.default_rng(42)
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    val_indices = set(indices[:n_val].tolist())
    train_windows = [w for i, w in enumerate(windows) if i not in val_indices]
    val_windows = [w for i, w in enumerate(windows) if i in val_indices]

    # Compute normalization on training set only (use max horizon)
    max_horizon = max(h for h, _, _ in curriculum_phases)
    norm = compute_normalization(train_windows, horizon_steps=max_horizon)
    np.savez(output_dir / "normalization.npz", **norm)

    torch_device = torch.device(device)
    best_ckpt_path = None
    best_val_loss = float("inf")
    history: list[dict] = []

    # TensorBoard setup
    writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(output_dir / "runs"))
        except ImportError:
            print("Warning: tensorboard not available, skipping TB logging")

    global_step = 0

    for phase_idx, (horizon, epochs, lr) in enumerate(curriculum_phases):
        print(f"\n=== Curriculum Phase {phase_idx + 1}/{len(curriculum_phases)}: H={horizon}, lr={lr} ===")

        # Filter and truncate windows to current horizon
        phase_train = [w for w in train_windows if w["controls"].shape[0] >= horizon]
        phase_val = [w for w in val_windows if w["controls"].shape[0] >= horizon]

        if not phase_train:
            raise ValueError(f"No training windows with horizon >= {horizon}")

        def truncate(w: dict) -> dict:
            return {
                "state": w["state"],
                "controls": w["controls"][:horizon],
                "costmap_grid": _window_costmap_grid(w),
                "goal_path_features": _window_goal_path_features(w),
                "target_states": w["target_states"][:horizon],
                "target_risk": w["target_risk"][:horizon],
            }

        phase_train = [truncate(w) for w in phase_train]
        phase_val = [truncate(w) for w in phase_val]

        train_dataset = SequenceFdmDataset(phase_train, horizon_steps=horizon)
        val_dataset = SequenceFdmDataset(phase_val, horizon_steps=horizon)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create or load model
        model = SequenceFdmMlpV2(horizon_steps=horizon, hidden_dims=hidden_dims).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Resume logic: load model + optimizer from previous training round
        if resume_from is not None:
            resume_path = Path(resume_from)
            resume_model_path = resume_path / "best_model.pt"
            resume_opt_path = resume_path / "optimizer.pt"
            if resume_model_path.exists():
                ckpt = torch.load(resume_model_path, map_location=device, weights_only=False)
                if ckpt.get("horizon_steps") == horizon:
                    model.load_state_dict(ckpt["model_state_dict"])
                    print(f"  Resumed model from {resume_model_path}")
                else:
                    print(f"  Warning: checkpoint horizon {ckpt.get('horizon_steps')} != current {horizon}, starting fresh")
            if resume_opt_path.exists():
                opt_ckpt = torch.load(resume_opt_path, map_location=device, weights_only=False)
                optimizer.load_state_dict(opt_ckpt["optimizer_state_dict"])
                print(f"  Resumed optimizer state from {resume_opt_path}")
            resume_from = None  # only resume on first phase
        elif best_ckpt_path is not None and best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            if ckpt.get("horizon_steps") == horizon:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"  Loaded checkpoint from phase {phase_idx}")
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        phase_best_loss = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_losses = []
            train_traj_losses = []
            train_risk_losses = []
            for state, controls, grid, goal_path_features, target_states, target_risk in train_loader:
                state = state.to(torch_device)
                controls = controls.to(torch_device)
                grid = grid.to(torch_device)
                goal_path_features = goal_path_features.to(torch_device)
                target_states = target_states.to(torch_device)
                target_risk = target_risk.to(torch_device)

                pred_states, pred_risk_logits = model(state, controls, grid, goal_path_features)

                loss_traj = mse_loss(pred_states, target_states)
                loss_risk = bce_loss(pred_risk_logits, target_risk)
                loss = w_traj * loss_traj + w_risk * loss_risk

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))
                train_traj_losses.append(float(loss_traj.item()))
                train_risk_losses.append(float(loss_risk.item()))
                global_step += 1

            # Validation
            model.eval()
            val_losses = []
            val_traj_losses = []
            val_risk_losses = []
            with torch.no_grad():
                for state, controls, grid, goal_path_features, target_states, target_risk in val_loader:
                    state = state.to(torch_device)
                    controls = controls.to(torch_device)
                    grid = grid.to(torch_device)
                    goal_path_features = goal_path_features.to(torch_device)
                    target_states = target_states.to(torch_device)
                    target_risk = target_risk.to(torch_device)

                    pred_states, pred_risk_logits = model(state, controls, grid, goal_path_features)
                    loss_traj = mse_loss(pred_states, target_states)
                    loss_risk = bce_loss(pred_risk_logits, target_risk)
                    loss = w_traj * loss_traj + w_risk * loss_risk
                    val_losses.append(float(loss.item()))
                    val_traj_losses.append(float(loss_traj.item()))
                    val_risk_losses.append(float(loss_risk.item()))

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_losses))
            avg_train_traj = float(np.mean(train_traj_losses))
            avg_train_risk = float(np.mean(train_risk_losses))
            avg_val_traj = float(np.mean(val_traj_losses))
            avg_val_risk = float(np.mean(val_risk_losses))

            print(f"  Epoch {epoch + 1}/{epochs}: train={avg_train:.6f}, val={avg_val:.6f}")

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar(f"Phase{phase_idx}/Loss/train", avg_train, epoch)
                writer.add_scalar(f"Phase{phase_idx}/Loss/val", avg_val, epoch)
                writer.add_scalar(f"Phase{phase_idx}/TrajLoss/train", avg_train_traj, epoch)
                writer.add_scalar(f"Phase{phase_idx}/TrajLoss/val", avg_val_traj, epoch)
                writer.add_scalar(f"Phase{phase_idx}/RiskLoss/train", avg_train_risk, epoch)
                writer.add_scalar(f"Phase{phase_idx}/RiskLoss/val", avg_val_risk, epoch)
                writer.add_scalar(f"Phase{phase_idx}/LearningRate", lr, epoch)

            history.append({
                "phase": phase_idx,
                "epoch": epoch,
                "horizon": horizon,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "train_traj_loss": avg_train_traj,
                "train_risk_loss": avg_train_risk,
                "val_traj_loss": avg_val_traj,
                "val_risk_loss": avg_val_risk,
            })

            if avg_val < phase_best_loss:
                phase_best_loss = avg_val
                no_improve = 0
                ckpt_path = output_dir / f"best_model_h{horizon}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "horizon_steps": horizon,
                    "hidden_dims": hidden_dims,
                    "input_dim": 6 + 3 * horizon + COSTMAP_GRID_DIM + GOAL_PATH_FEATURE_DIM,
                    "target_dim": 6 * horizon + horizon,
                    "architecture": "dual_branch_costmap_cnn_v1",
                    "phase": phase_idx,
                }, ckpt_path)
                best_ckpt_path = ckpt_path
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "horizon_steps": horizon,
                        "hidden_dims": hidden_dims,
                        "input_dim": 6 + 3 * horizon + COSTMAP_GRID_DIM + GOAL_PATH_FEATURE_DIM,
                        "target_dim": 6 * horizon + horizon,
                        "architecture": "dual_branch_costmap_cnn_v1",
                        "phase": phase_idx,
                    }, output_dir / "best_model.pt")
                    torch.save({
                        "optimizer_state_dict": optimizer.state_dict(),
                        "horizon_steps": horizon,
                        "lr": lr,
                    }, output_dir / "optimizer.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    if writer is not None:
        writer.close()

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "output_dir": str(output_dir),
    }

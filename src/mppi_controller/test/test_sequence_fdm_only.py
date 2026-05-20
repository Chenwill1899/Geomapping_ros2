from __future__ import annotations

import pytest


def test_sequence_fdm_model_does_not_export_old_residual_api() -> None:
    import mppi_controller.core.sequence_fdm_model as model

    assert hasattr(model, "SequenceFdmMlp")
    assert hasattr(model, "sequence_fdm_feature_names")
    assert not hasattr(model, "ResidualFdmMlp")
    assert not hasattr(model, "build_feature_vector")


def test_train_command_is_sequence_fdm_by_default() -> None:
    from mppi_controller.cli import build_parser

    args = build_parser().parse_args(["train", "--dataset", "splits", "--output", "out"])

    assert args.handler == "train"
    assert not hasattr(args, "sequence")
    assert args.sequence_horizon == 25


def test_enabled_fdm_rejects_old_residual_mode_without_loading_artifacts() -> None:
    from mppi_controller.simulation.omni_runner import create_omni_controller

    config = {
        "simulation": {"sampling_rate": 10.0, "time_horizon": 1.0},
        "mppi": {"backend": "torch"},
        "robot": {},
        "fdm": {"enabled": True, "mode": "residual", "model_dir": "missing"},
    }

    with pytest.raises(ValueError, match="Only sequence FDM"):
        create_omni_controller(config)


def test_enabled_fdm_rejects_numpy_backend() -> None:
    from mppi_controller.simulation.omni_runner import create_omni_controller

    config = {
        "mppi": {"backend": "numpy"},
        "fdm": {"enabled": True, "model_dir": "missing"},
    }

    with pytest.raises(ValueError, match="sequence FDM requires torch"):
        create_omni_controller(config)

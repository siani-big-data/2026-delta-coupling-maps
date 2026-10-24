from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.dataset import validate_model_input
from ..models import build_model, trainable_parameter_count
from ..models.specs import MODEL_INTERFACES


@dataclass(frozen=True)
class _SeedInputs:
    inputs_numpy: np.ndarray
    covariates_numpy: np.ndarray
    targets_numpy: np.ndarray
    split_values: np.ndarray
    record_ids: np.ndarray
    masks: dict[str, np.ndarray]
    inputs: torch.Tensor
    covariates: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_artifact(cls, name: str, artifact: dict[str, np.ndarray]) -> _SeedInputs:
        inputs_numpy = np.asarray(artifact["inputs"], dtype=np.float32)
        covariates_numpy = np.asarray(artifact["covariates"], dtype=np.float32)
        targets_numpy = np.asarray(artifact["target"], dtype=np.float32)
        split_values = artifact["split"].astype(str)
        record_ids = artifact["record_id"].astype(str)
        validate_model_input(name, inputs_numpy)
        if not np.isfinite(inputs_numpy).all() or not np.isfinite(covariates_numpy).all():
            raise ValueError(f"Non-finite model inputs remain in {name}.npz")
        if not np.isfinite(targets_numpy).all():
            raise ValueError(f"Non-finite targets remain in {name}.npz")
        masks = {split: split_values == split for split in ("train", "validation", "test")}
        if any(not mask.any() for mask in masks.values()):
            raise ValueError(f"{name}.npz must contain train, validation, and test records")
        return cls(
            inputs_numpy=inputs_numpy,
            covariates_numpy=covariates_numpy,
            targets_numpy=targets_numpy,
            split_values=split_values,
            record_ids=record_ids,
            masks=masks,
            inputs=torch.from_numpy(inputs_numpy),
            covariates=torch.from_numpy(covariates_numpy),
            targets=torch.from_numpy(targets_numpy),
        )

    def train_loader(self, seed: int, batch_size: int) -> DataLoader:
        indices = np.flatnonzero(self.masks["train"])
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            TensorDataset(self.inputs[indices], self.covariates[indices], self.targets[indices]),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    residuals = predictions - targets
    return {
        "count": int(targets.size),
        "mae_years": float(np.mean(np.abs(residuals))),
        "rmse_years": float(np.sqrt(np.mean(np.square(residuals)))),
    }


def _predict(
    model: nn.Module,
    inputs: torch.Tensor,
    covariates: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(inputs, covariates), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_inputs, batch_covariates in loader:
            prediction = model(batch_inputs.to(device), batch_covariates.to(device))
            output.append(prediction.detach().cpu().numpy())
    return np.concatenate(output).astype(float)


def _train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    loss_total = 0.0
    seen = 0
    for inputs, covariates, targets in loader:
        inputs = inputs.to(device)
        covariates = covariates.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs, covariates), targets)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.detach().cpu()) * targets.size(0)
        seen += targets.size(0)
    return loss_total / seen


def _split_mae(
    model: nn.Module,
    data: _SeedInputs,
    split: str,
    device: torch.device,
    batch_size: int,
) -> float:
    mask = data.masks[split]
    predictions = _predict(model, data.inputs[mask], data.covariates[mask], device, batch_size)
    return float(_metrics(data.targets_numpy[mask], predictions)["mae_years"])


def _fit_epochs(
    model: nn.Module,
    data: _SeedInputs,
    loader: DataLoader,
    *,
    name: str,
    seed: int,
    device: torch.device,
    epochs_maximum: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[dict[str, float | int]], int, dict[str, torch.Tensor]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    best_validation_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    log_interval = max(5, epochs_maximum // 50)

    for epoch in range(1, epochs_maximum + 1):
        training_loss = _train_epoch(model, loader, optimizer, loss_function, device)
        validation_mae = _split_mae(model, data, "validation", device, batch_size)
        training_mae = _split_mae(model, data, "train", device, batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_mse_loss": training_loss,
                "train_mae_years": training_mae,
                "validation_mae_years": validation_mae,
            }
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[TRAIN] model={name} seed={seed} early_stop epoch={epoch} "
                    f"best_epoch={best_epoch} best_validation_mae={best_validation_mae:.4f}",
                    flush=True,
                )
                break
        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs_maximum:
            print(
                f"[EPOCH] model={name} seed={seed} epoch={epoch}/{epochs_maximum} "
                f"train_mae={training_mae:.4f} "
                f"validation_mae={validation_mae:.4f} best={best_validation_mae:.4f}",
                flush=True,
            )

    assert best_state is not None
    return history, best_epoch, best_state


def _train_one_seed(
    name: str,
    artifact: dict[str, np.ndarray],
    seed_dir: Path,
    *,
    seed: int,
    device: torch.device,
    epochs_maximum: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    _set_training_seed(seed)
    data = _SeedInputs.from_artifact(name, artifact)
    train_loader = data.train_loader(seed, batch_size)
    input_shape = list(data.inputs_numpy.shape[1:])
    model_config = {"name": name, "input_shape": input_shape}
    model = build_model(name, tuple(input_shape)).to(device)
    print(
        f"[TRAIN] model={name} seed={seed} device={device} "
        f"train={data.masks['train'].sum()} validation={data.masks['validation'].sum()} "
        f"test={data.masks['test'].sum()}",
        flush=True,
    )
    history, best_epoch, best_state = _fit_epochs(
        model,
        data,
        train_loader,
        name=name,
        seed=seed,
        device=device,
        epochs_maximum=epochs_maximum,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.load_state_dict(best_state)
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": name,
            "seed": seed,
            "best_epoch": best_epoch,
            "model_config": model_config,
            "state_dict": best_state,
        },
        seed_dir / "best.pt",
    )
    pd.DataFrame(history).to_csv(seed_dir / "history.csv", index=False)

    final_predictions = _predict(model, data.inputs, data.covariates, device, batch_size)
    predictions = pd.DataFrame(
        {
            "record_id": data.record_ids,
            "split": data.split_values,
            "target": data.targets_numpy.astype(float),
            "prediction": final_predictions,
            "absolute_error": np.abs(final_predictions - data.targets_numpy),
        }
    )
    predictions.to_csv(seed_dir / "predictions.csv", index=False)
    split_metrics = {
        split: _metrics(data.targets_numpy[mask], final_predictions[mask])
        for split, mask in data.masks.items()
    }
    metadata: dict[str, object] = {
        "model": name,
        "architecture_interface": MODEL_INTERFACES[name],
        "model_config": model_config,
        "publishable_result": "requires full-cohort protocol review",
        "weights_updated": True,
        "seed": seed,
        "device": str(device),
        "input_shape": input_shape,
        "record_count": int(data.targets_numpy.size),
        "trainable_parameters": trainable_parameter_count(model),
        "best_epoch": best_epoch,
        "stopped_early": len(history) < epochs_maximum,
        "training_configuration": {
            "epochs_maximum": epochs_maximum,
            "early_stopping_patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "metrics": split_metrics,
    }
    (seed_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[TRAIN COMPLETE] model={name} seed={seed} best_epoch={best_epoch} "
        f"test_mae={float(split_metrics['test']['mae_years']):.4f}",
        flush=True,
    )
    return predictions, metadata

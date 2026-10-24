from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baroreflex_pipeline.models import build_model


def load_trained_model(checkpoint: Path, *, metadata_path: Path | None = None):
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata_path = metadata_path or checkpoint.with_name("run.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    configuration = payload.get("model_config") or metadata.get("model_config")
    if configuration is None:
        configuration = {
            "name": metadata.get("variant", payload.get("model")),
            "input_shape": metadata.get("input_shape"),
        }
    if not configuration.get("name"):
        raise ValueError("Checkpoint has no model identity; supply its original run.json")
    shape = tuple(configuration["input_shape"]) if configuration.get("input_shape") else None
    model = build_model(configuration["name"], shape)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, configuration


def predict_checkpoint(
    checkpoint: Path, inputs_path: Path, output: Path, *, device: str = "auto", batch_size: int = 64
) -> pd.DataFrame:
    from .engine import _predict, _resolve_device

    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model, configuration = load_trained_model(checkpoint)
    with np.load(inputs_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    x = np.asarray(data["inputs"], dtype=np.float32)
    covariates = np.asarray(data["covariates"], dtype=np.float32)
    ids = data["record_id"].astype(str)
    if (
        len(x) == 0
        or covariates.shape != (len(x), 4)
        or ids.shape != (len(x),)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Misaligned or duplicate participant inputs")
    if not np.isfinite(x).all() or not np.isfinite(covariates).all():
        raise ValueError("Non-finite prepared inputs")
    if configuration.get("input_shape") and list(x.shape[1:]) != list(configuration["input_shape"]):
        raise ValueError("Input shape does not match the checkpoint")
    resolved = _resolve_device(device)
    values = _predict(
        model.to(resolved), torch.from_numpy(x), torch.from_numpy(covariates), resolved, batch_size
    )
    frame = pd.DataFrame({"record_id": ids})
    for key in ("split", "target"):
        if key in data:
            if data[key].shape != (len(x),):
                raise ValueError(f"Misaligned {key}")
            frame[key] = data[key]
    frame["prediction"] = values
    if "target" in frame:
        frame["absolute_error"] = abs(frame["prediction"] - frame["target"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame

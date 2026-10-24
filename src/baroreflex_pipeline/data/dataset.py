from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.specs import MODEL_INPUTS
from ..utils.provenance import sha256_file


def validate_model_input(name: str, inputs: np.ndarray) -> None:
    shape = inputs.shape[1:]
    if MODEL_INPUTS[name] == "dcm":
        valid = len(shape) == 3 and shape[0] == 2 and min(shape[1:]) >= 4
    else:
        valid = len(shape) == 1 and shape[0] > 0
    if not valid:
        raise ValueError(f"Unexpected input shape for {name}: {shape}")


def check_predictions(frame: pd.DataFrame, artifact: dict[str, np.ndarray]) -> None:
    required = {"record_id", "split", "target", "prediction"}
    if not required.issubset(frame.columns) or len(frame) != len(artifact["target"]):
        raise ValueError("Incomplete participant predictions")
    for key in ("record_id", "split", "target"):
        if key == "target":
            actual = frame[key].to_numpy(dtype=float)
            expected = np.asarray(artifact[key], dtype=np.float32).astype(float)
            equal = np.allclose(actual, expected, rtol=0, atol=1e-10)
        else:
            equal = np.array_equal(frame[key].to_numpy(dtype=str), artifact[key].astype(str))
        if not equal:
            raise ValueError(f"Prediction identity mismatch: {key}")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite predictions")


@dataclass
class PreparedInputs:
    artifacts: dict[str, dict[str, np.ndarray]]
    checksums: dict[str, str]

    @classmethod
    def load(cls, directory: Path, models: tuple[str, ...]):
        artifacts, checksums = {}, {}
        reference = None
        for name in sorted({MODEL_INPUTS[name] for name in models}):
            path = directory / f"{name}.npz"
            with np.load(path, allow_pickle=False) as source:
                artifact = {key: source[key] for key in source.files}
            cls._validate_artifact(name, artifact)
            identity = tuple(
                artifact[key] for key in ("record_id", "split", "target", "covariates")
            )
            if reference is not None and not all(
                np.array_equal(a, b) for a, b in zip(reference, identity)
            ):
                raise ValueError(f"Subjects, splits, targets, or covariates differ in {name}.npz")
            reference = identity
            artifacts[name] = artifact
            checksums[path.name] = sha256_file(path)
        return cls(artifacts, checksums)

    @staticmethod
    def _validate_artifact(name, artifact):
        required = {"inputs", "covariates", "target", "record_id", "split"}
        if not required.issubset(artifact):
            raise ValueError(f"{name}.npz is missing required arrays")
        count = len(artifact["target"])
        if any(len(artifact[key]) != count for key in required) or artifact["covariates"].shape != (
            count,
            4,
        ):
            raise ValueError(f"Misaligned tensors in {name}.npz")
        if len(np.unique(artifact["record_id"])) != count or set(artifact["split"].astype(str)) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"Invalid participant IDs or partitions in {name}.npz")
        validate_model_input(name, artifact["inputs"])
        if not all(np.isfinite(artifact[key]).all() for key in ("inputs", "covariates", "target")):
            raise ValueError(f"Non-finite tensors in {name}.npz")

    def for_model(self, name: str) -> dict[str, np.ndarray]:
        return self.artifacts[MODEL_INPUTS[name]]

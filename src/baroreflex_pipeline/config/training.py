from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models.specs import MODEL_INPUTS
from .experiment import load_experiment_config


@dataclass(frozen=True)
class ExperimentSettings:
    specification: dict
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    fit: dict

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        models=None,
        seeds=None,
        epochs_maximum=None,
        patience=None,
        batch_size=None,
        learning_rate=None,
        weight_decay=None,
    ):
        specification = load_experiment_config(path)
        training = specification["training"]
        names = tuple(models if models is not None else training["models"])
        selected_seeds = tuple(seeds if seeds is not None else training["run_seeds"])
        cls._validate_models(names)
        cls._validate_seeds(selected_seeds)
        if training["optimizer"] != "AdamW" or training["loss"] != "mean_squared_error":
            raise ValueError("Supported optimizer/loss: AdamW and mean_squared_error")
        overrides = {
            "epochs_maximum": epochs_maximum,
            "early_stopping_patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        fit = {
            key: value if value is not None else training[key] for key, value in overrides.items()
        }
        cls._validate_fit(fit)
        return cls(specification, names, selected_seeds, fit)

    @staticmethod
    def _validate_models(models):
        if not models or len(set(models)) != len(models) or set(models).difference(MODEL_INPUTS):
            raise ValueError(f"Choose unique model names from {', '.join(MODEL_INPUTS)}")

    @staticmethod
    def _validate_seeds(seeds):
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be a nonempty list of unique integers")
        if any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("Seeds must be integers in [0, 2**32)")

    @staticmethod
    def _validate_fit(fit):
        for key in ("epochs_maximum", "early_stopping_patience", "batch_size"):
            if not isinstance(fit[key], int) or fit[key] < 1:
                raise ValueError("Epochs, patience, and batch size must be positive integers")
        values = [fit["learning_rate"], fit["weight_decay"]]
        if not np.isfinite(values).all() or values[0] <= 0 or values[1] < 0:
            raise ValueError("Learning rate must be positive and weight decay nonnegative")

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class Trainer(ABC):
    @abstractmethod
    def fit(
        self, training_dir: Path, run_dir: Path, *, experiment_config: Path, **settings
    ) -> pd.DataFrame:

        raise NotImplementedError


class TorchTrainer(Trainer):
    name = "torch"

    def fit(
        self, training_dir: Path, run_dir: Path, *, experiment_config: Path, **settings
    ) -> pd.DataFrame:
        from .runner import run_experiment

        return run_experiment(
            training_dir, run_dir, experiment_config=experiment_config, **settings
        )

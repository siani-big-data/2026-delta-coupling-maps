from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.training import ExperimentSettings
from ..data.dataset import PreparedInputs, check_predictions
from ..utils.provenance import experiment_lock, sha256_file, write_json
from . import engine


class ExperimentRunner:
    def __init__(
        self, training_dir: Path, run_dir: Path, settings: ExperimentSettings, device: str
    ):
        self.inputs = PreparedInputs.load(Path(training_dir), settings.models)
        self.run_dir = Path(run_dir)
        self.settings = settings
        self.device = engine._resolve_device(device)
        self.completed = 0
        self.total = len(settings.models) * len(settings.seeds)
        self.started = time.monotonic()
        self.manifest = self._create_manifest()
        self.summary = {
            "status": "running",
            "weights_updated": True,
            "device": str(self.device),
            "seeds": list(settings.seeds),
            "configuration": settings.fit,
            "models": {},
        }

    def run(self) -> pd.DataFrame:
        with experiment_lock(self.run_dir):
            self._validate_resume()
            self.manifest["status"] = "running"
            write_json(self.run_dir / "run_manifest.json", self.manifest)
            try:
                combined = self._fit_models()
                combined.to_csv(self.run_dir / "paired_predictions.csv", index=False)
                self.summary["status"] = self.manifest["status"] = "complete"
                print(f"[COMPLETE] {self.run_dir}", flush=True)
                return combined
            except BaseException as error:
                self.summary.update(status="interrupted_or_failed", error=repr(error))
                self.manifest.update(status="interrupted_or_failed", error=repr(error))
                raise
            finally:
                self._save_state()

    def _create_manifest(self) -> dict:
        package = Path(__file__).resolve().parents[1]
        return {
            "format_version": 1,
            "models": list(self.settings.models),
            "seeds": list(self.settings.seeds),
            "training_configuration": self.settings.fit,
            "experiment_configuration": self.settings.specification,
            "input_sha256": self.inputs.checksums,
            "source_sha256": {
                str(path.relative_to(package)): sha256_file(path)
                for path in sorted(package.rglob("*.py"))
            },
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("numpy", "pandas", "torch", "scikit-learn")
            },
            "device": str(self.device),
        }

    def _validate_resume(self) -> None:
        manifest_path = self.run_dir / "run_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if {key: previous.get(key) for key in self.manifest} != self.manifest:
                raise ValueError(
                    "Run inputs, source, environment, or configuration changed; choose a new --run-dir"
                )
        elif any(path.name != ".run.lock" for path in self.run_dir.iterdir()):
            raise ValueError(
                "Run directory contains untracked/legacy output; choose a new --run-dir"
            )

    def _fit_models(self) -> pd.DataFrame:
        combined = None
        for name in self.settings.models:
            predictions = self._fit_model(name)
            frame = predictions[["record_id", "split", "target"]].copy()
            frame[f"{name}_prediction"] = predictions["prediction"]
            combined = (
                frame
                if combined is None
                else combined.merge(
                    frame, on=["record_id", "split", "target"], validate="one_to_one"
                )
            )
            self._save_state()
        return combined

    def _fit_model(self, name: str) -> pd.DataFrame:
        artifact = self.inputs.for_model(name)
        frames, metadata = [], []
        for seed in self.settings.seeds:
            frame, seed_metadata = self._fit_seed(name, seed, artifact)
            check_predictions(frame, artifact)
            frames.append(frame)
            metadata.append(seed_metadata)
            self._report_progress()
        ensemble = self._ensemble(frames)
        summary = {
            "variant": name,
            "trainable_parameters": metadata[0]["trainable_parameters"],
            "seed_runs": metadata,
            "ensemble_strategy": "mean_prediction_across_seeds",
            "ensemble_metrics": self._ensemble_metrics(ensemble),
        }
        ensemble.to_csv(self.run_dir / name / "predictions.csv", index=False)
        write_json(self.run_dir / name / "summary.json", summary)
        self.summary["models"][name] = summary
        return ensemble

    def _fit_seed(self, name: str, seed: int, artifact: dict):
        directory = self.run_dir / name / f"seed_{seed}"
        print(
            f"[EXPERIMENT] model={name} seed={seed} "
            f"run={self.completed + 1}/{self.total} device={self.device}",
            flush=True,
        )
        artifacts = ("run.json", "best.pt", "predictions.csv", "history.csv")
        if all((directory / filename).is_file() for filename in artifacts):
            return self._resume_seed(directory, name, seed)
        settings = self.settings.fit
        return engine._train_one_seed(
            name,
            artifact,
            directory,
            seed=seed,
            device=self.device,
            epochs_maximum=settings["epochs_maximum"],
            patience=settings["early_stopping_patience"],
            batch_size=settings["batch_size"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )

    def _resume_seed(self, directory: Path, name: str, seed: int):
        metadata = json.loads((directory / "run.json").read_text())
        expected = {
            "training_configuration": self.settings.fit,
            "model": name,
            "seed": seed,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Incompatible completed seed: {directory}")
        predictions = pd.read_csv(directory / "predictions.csv", dtype={"record_id": str})
        print(f"[RESUME] model={name} seed={seed}", flush=True)
        return predictions, metadata

    def _report_progress(self) -> None:
        self.completed += 1
        elapsed = (time.monotonic() - self.started) / 60
        print(
            f"[PROGRESS] completed={self.completed}/{self.total} elapsed_minutes={elapsed:.1f}",
            flush=True,
        )

    def _save_state(self) -> None:
        write_json(self.run_dir / "summary.json", self.summary)
        write_json(self.run_dir / "run_manifest.json", self.manifest)

    @staticmethod
    def _ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
        result = frames[0][["record_id", "split", "target"]].copy()
        matrix = np.column_stack([frame["prediction"].to_numpy(float) for frame in frames])
        result["prediction"] = matrix.mean(axis=1)
        result["prediction_standard_deviation"] = matrix.std(axis=1, ddof=0)
        result["absolute_error"] = abs(result["prediction"] - result["target"])
        return result

    @staticmethod
    def _ensemble_metrics(ensemble: pd.DataFrame) -> dict:
        return {
            split: engine._metrics(
                group["target"].to_numpy(float), group["prediction"].to_numpy(float)
            )
            for split, group in ensemble.groupby("split", sort=True)
        }


def run_experiment(
    training_dir: Path,
    run_dir: Path,
    *,
    experiment_config: Path,
    seeds: list[int] | None = None,
    epochs_maximum: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    device: str = "auto",
    models: list[str] | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> pd.DataFrame:
    settings = ExperimentSettings.from_file(
        experiment_config,
        models=models,
        seeds=seeds,
        epochs_maximum=epochs_maximum,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return ExperimentRunner(training_dir, run_dir, settings, device).run()

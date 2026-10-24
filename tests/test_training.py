import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from baroreflex_pipeline.config import PipelineConfig
from baroreflex_pipeline.training import (
    prepare_training_artifacts,
    run_real_training,
)
from baroreflex_pipeline.workflows.experiment import run_complete_workflow
from tests.support import EXPERIMENT_CONFIG, PROCESSING_CONFIG


class TrainingPipelineTests(unittest.TestCase):
    def _synthetic_outputs(self, root: Path) -> None:
        record_ids = [f"{index:04d}" for index in range(1, 7)]
        splits = pd.DataFrame(
            {
                "record_id": record_ids,
                "status": "ok",
                "Age_group": [1, 2, 3, 4, 5, 6],
                "Sex": [0, 1, 0, 1, 0, 1],
                "BMI": [20, 22, 24, 26, 28, 30],
                "Length": [15, 15, 20, 20, 25, 25],
                "Device": [0, 1, 0, 1, 0, 1],
                "age_midpoint_years": [18.5, 22, 27, 32, 37, 42],
                "split": ["train", "train", "validation", "validation", "test", "test"],
            }
        )
        splits.to_csv(root / "splits.csv", index=False)
        clinical = pd.DataFrame(
            {
                "record_id": record_ids,
                "indicator_a": [1, 2, 3, 4, 5, 6],
                "indicator_b": [1, np.nan, 3, 4, 5, 6],
            }
        )
        clinical.to_csv(root / "clinical_indicators.csv", index=False)
        scalar = pd.DataFrame(
            {
                "record_id": record_ids,
                "feature_a": [10, 20, 30, 40, 50, 60],
                "feature_b": [2, 4, 6, 8, 10, 12],
            }
        )
        scalar.to_csv(root / "scalar_features.csv", index=False)
        (root / "dcm").mkdir()
        for index, record_id in enumerate(record_ids):
            np.savez_compressed(
                root / "dcm" / f"{record_id}.npz", dcm=np.full((2, 25, 25), index, dtype=np.float32)
            )

    def test_prepared_representations_are_aligned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._synthetic_outputs(root)

            metadata = prepare_training_artifacts(root)

            self.assertEqual(metadata["record_count"], 6)
            self.assertEqual(set(metadata["representations"]), {"clinical", "scalar", "dcm"})
            with np.load(root / "training_data" / "clinical.npz") as source:
                reference = {key: source[key] for key in source.files}
            for name in ("clinical", "scalar", "dcm"):
                with np.load(root / "training_data" / f"{name}.npz") as artifact:
                    for key in ("record_id", "split", "target", "covariates"):
                        np.testing.assert_array_equal(artifact[key], reference[key])
                    self.assertTrue(np.isfinite(artifact["inputs"]).all())
                    if name != "dcm":
                        train = artifact["split"].astype(str) == "train"
                        np.testing.assert_allclose(artifact["inputs"][train].mean(axis=0), 0.0)

    def test_complete_workflow_uses_real_trainer(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as patches:
            root = Path(temporary)
            for operation in (
                "process_dataset",
                "save_subject_splits",
                "audit_excluded_records",
                "audit_rr_morphology",
                "prepare_training_artifacts",
            ):
                patches.enter_context(
                    patch(f"baroreflex_pipeline.workflows.experiment.{operation}")
                )
            fit = patches.enter_context(
                patch("baroreflex_pipeline.training.backends.TorchTrainer.fit")
            )
            result = run_complete_workflow(
                root / "raw",
                root / "outputs",
                PipelineConfig.from_json(PROCESSING_CONFIG),
                experiment_config=EXPERIMENT_CONFIG,
                download=False,
                training_device="cpu",
                training_epochs=1,
            )
            fit.assert_called_once()
            self.assertEqual(
                fit.call_args.args, (root / "outputs/training_data", root / "outputs/runs/torch")
            )
            self.assertEqual(fit.call_args.kwargs["device"], "cpu")
            self.assertEqual(fit.call_args.kwargs["epochs_maximum"], 1)
            self.assertEqual(result["trainer"], "torch")
            self.assertEqual(result["stages"][-1]["stage"], "torch_training")

    def test_real_training_writes_checkpoints_and_ensemble_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_dir = root / "training_data"
            training_dir.mkdir()
            random = np.random.default_rng(2025)
            record_ids = np.asarray([f"{index:04d}" for index in range(1, 13)])
            splits = np.asarray(["train"] * 6 + ["validation"] * 3 + ["test"] * 3)
            targets = np.linspace(20, 75, 12, dtype=np.float32)
            covariates = random.normal(size=(12, 4)).astype(np.float32)
            representations = {
                "clinical": random.normal(size=(12, 21)).astype(np.float32),
                "scalar": random.normal(size=(12, 3880)).astype(np.float32),
                "dcm": random.normal(size=(12, 2, 25, 25)).astype(np.float32),
            }
            for name, inputs in representations.items():
                np.savez_compressed(
                    training_dir / f"{name}.npz",
                    inputs=inputs,
                    record_id=record_ids,
                    split=splits,
                    target=targets,
                    covariates=covariates,
                )

            run_dir = root / "training_runs" / "torch"
            predictions = run_real_training(
                training_dir,
                run_dir,
                experiment_config=EXPERIMENT_CONFIG,
                seeds=[7],
                epochs_maximum=1,
                patience=1,
                batch_size=4,
                device="cpu",
            )

            self.assertEqual(len(predictions), 12)
            self.assertEqual(
                set(predictions.columns),
                {
                    "record_id",
                    "split",
                    "target",
                    "clinical_prediction",
                    "scalar_prediction",
                    "dcm_prediction",
                },
            )
            self.assertTrue(np.isfinite(predictions.filter(like="_prediction").to_numpy()).all())
            for name in representations:
                self.assertTrue((run_dir / name / "seed_7" / "best.pt").exists())
                self.assertTrue((run_dir / name / "predictions.csv").exists())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertTrue(summary["weights_updated"])

            with patch(
                "baroreflex_pipeline.training.engine._train_one_seed",
                side_effect=AssertionError("completed seed should have been reused"),
            ):
                resumed = run_real_training(
                    training_dir,
                    run_dir,
                    experiment_config=EXPERIMENT_CONFIG,
                    seeds=[7],
                    epochs_maximum=1,
                    patience=1,
                    batch_size=4,
                    device="cpu",
                )
            self.assertEqual(len(resumed), 12)


if __name__ == "__main__":
    unittest.main()

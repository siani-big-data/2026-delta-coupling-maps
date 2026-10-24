import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from baroreflex_pipeline.cli import main
from baroreflex_pipeline.data.preparation import _standardize_from_training
from baroreflex_pipeline.models import ABLATION_SPECS, MODEL_INPUTS, build_model
from baroreflex_pipeline.training import run_real_training
from baroreflex_pipeline.training.checkpoints import load_trained_model, predict_checkpoint
from baroreflex_pipeline.utils.provenance import experiment_lock, sha256_file
from tests.support import EXPERIMENT_CONFIG


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        generator = np.random.default_rng(3)
        self.shared = {
            "record_id": np.array([f"{i:04d}" for i in range(12)]),
            "split": np.array(["train"] * 6 + ["validation"] * 3 + ["test"] * 3),
            "target": np.linspace(20, 60, 12, dtype=np.float32),
            "covariates": generator.normal(size=(12, 4)).astype(np.float32),
        }

        for name, shape in (("clinical", (3,)), ("scalar", (17,)), ("dcm", (2, 12, 16))):
            np.savez_compressed(
                self.inputs / f"{name}.npz",
                inputs=generator.normal(size=(12, *shape)).astype(np.float32),
                **self.shared,
            )

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_models_checkpoint_roundtrip_and_resume(self):
        run = self.root / "run"
        arguments = dict(
            experiment_config=EXPERIMENT_CONFIG,
            models=list(MODEL_INPUTS),
            seeds=[7],
            epochs_maximum=2,
            patience=2,
            batch_size=4,
            device="cpu",
        )
        result = run_real_training(self.inputs, run, **arguments)
        self.assertEqual(len(result), 12)
        for name, route in MODEL_INPUTS.items():
            checkpoint = run / name / "seed_7/best.pt"
            model, configuration = load_trained_model(checkpoint)
            self.assertEqual(configuration["name"], name)
            prediction = predict_checkpoint(
                checkpoint,
                self.inputs / f"{route}.npz",
                self.root / f"{name}.csv",
                device="cpu",
                batch_size=4,
            )
            saved = pd.read_csv(checkpoint.with_name("predictions.csv"))
            np.testing.assert_allclose(
                prediction["prediction"], saved["prediction"], rtol=0, atol=1e-6
            )
            self.assertFalse(model.training)
        with patch(
            "baroreflex_pipeline.training.engine._train_one_seed",
            side_effect=AssertionError("Unexpected retraining"),
        ):
            resumed = run_real_training(self.inputs, run, **arguments)
        pd.testing.assert_frame_equal(result, resumed)
        self.assertEqual(json.loads((run / "run_manifest.json").read_text())["status"], "complete")

    def test_resume_refuses_changed_data_and_settings(self):
        run = self.root / "run"
        args = dict(
            experiment_config=EXPERIMENT_CONFIG,
            models=["clinical"],
            seeds=[7],
            epochs_maximum=1,
            patience=1,
            device="cpu",
        )
        run_real_training(self.inputs, run, **args)
        with self.assertRaisesRegex(ValueError, "changed"):
            run_real_training(self.inputs, run, **{**args, "epochs_maximum": 2})
        with np.load(self.inputs / "clinical.npz") as source:
            artifact = {key: source[key] for key in source.files}
        artifact["inputs"][0, 0] += 1
        np.savez_compressed(self.inputs / "clinical.npz", **artifact)
        with self.assertRaisesRegex(ValueError, "changed"):
            run_real_training(self.inputs, run, **args)

    def test_resume_rejects_reordered_predictions(self):
        run = self.root / "run"
        args = dict(
            experiment_config=EXPERIMENT_CONFIG,
            models=["clinical"],
            seeds=[7],
            epochs_maximum=1,
            patience=1,
            device="cpu",
        )
        run_real_training(self.inputs, run, **args)
        path = run / "clinical/seed_7/predictions.csv"
        frame = pd.read_csv(path, dtype={"record_id": str}).iloc[::-1]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            run_real_training(self.inputs, run, **args)

    def test_lock_and_legacy_output_protection(self):
        run = self.root / "run"
        with experiment_lock(run):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                with experiment_lock(run):
                    self.fail("Concurrent writer accepted")
        (run / "summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "untracked/legacy"):
            run_real_training(
                self.inputs,
                run,
                experiment_config=EXPERIMENT_CONFIG,
                models=["clinical"],
                seeds=[7],
                epochs_maximum=1,
                patience=1,
                device="cpu",
            )

    def test_train_only_imputation_and_all_missing_column(self):
        values = np.array([[1, np.nan], [3, np.inf], [np.nan, 100], [9999, 200]])
        x, metadata = _standardize_from_training(
            values, np.array([True, True, False, False]), impute_missing=True
        )
        self.assertEqual(metadata["training_medians"], [2, 0])
        np.testing.assert_array_equal(x[:, 1], 0)
        self.assertEqual(x[2, 0], 0)
        self.assertEqual(x[3, 0], 9997)

    def test_omitted_inputs_and_channels_cannot_affect_predictions(self):
        covariates = torch.randn(2, 4)
        for name, (route, physiology, cov, channels) in ABLATION_SPECS.items():
            shape = {"clinical": (3,), "scalar": (17,), "dcm": (2, 12, 16)}[route]
            x = torch.randn(2, *shape)
            model = build_model(name, shape).eval()
            with torch.no_grad():
                expected = model(x, covariates)
                if not cov:
                    torch.testing.assert_close(expected, model(x, covariates + 100))
                    self.assertIsNone(model.covariates)
                if physiology == "covariates":
                    torch.testing.assert_close(expected, model(x + 100, covariates))
                    self.assertIsNone(model.physiology)
                if len(channels) == 1:
                    changed = x.clone()
                    changed[:, 1 - channels[0]] += 100
                    torch.testing.assert_close(expected, model(changed, covariates))

    def test_legacy_variant_identity_comes_from_metadata(self):
        model = build_model("covariates_only", (3,))
        checkpoint = self.root / "best.pt"
        torch.save({"model": "clinical", "state_dict": model.state_dict()}, checkpoint)
        checkpoint.with_name("run.json").write_text(
            json.dumps({"variant": "covariates_only", "input_shape": [3]})
        )
        _, configuration = load_trained_model(checkpoint)
        self.assertEqual(configuration["name"], "covariates_only")

    def test_cli_custom_training_and_portable_digest(self):
        with patch("baroreflex_pipeline.training.backends.TorchTrainer.fit") as train:
            main(
                [
                    "train",
                    "--training-dir",
                    str(self.inputs),
                    "--run-dir",
                    str(self.root / "cli"),
                    "--experiment-config",
                    str(EXPERIMENT_CONFIG),
                    "--models",
                    "clinical_only",
                    "--epochs",
                    "2",
                    "--learning-rate",
                    "0.02",
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(train.call_args.kwargs["models"], ["clinical_only"])
        self.assertEqual(train.call_args.kwargs["learning_rate"], 0.02)
        path = self.inputs / "clinical.npz"
        self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

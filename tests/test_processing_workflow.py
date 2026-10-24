import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baroreflex_pipeline.config import PipelineConfig
from baroreflex_pipeline.processing.signals import RecordResult
from baroreflex_pipeline.workflows.processing import process_dataset
from tests.support import PROCESSING_CONFIG
from tests.test_scalars import beat_table


class ProcessingWorkflowTests(unittest.TestCase):
    def test_feature_extractors_write_panels_and_provenance_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "0001.hea").write_text("synthetic")
            record = RecordResult("0001", beat_table(), 1000, "ECG", "NIBP")
            with patch(
                "baroreflex_pipeline.workflows.processing.extract_beat_series", return_value=record
            ):
                manifest = process_dataset(
                    raw,
                    root / "result",
                    PipelineConfig.from_json(PROCESSING_CONFIG),
                    scalar_feature_set="minimal",
                    scalar_n_jobs=1,
                    save_figures=False,
                )
            self.assertEqual(manifest.status.tolist(), ["ok"])
            provenance = json.loads((root / "result/provenance.json").read_text())
            self.assertGreater(provenance["scalar_feature_count"], 20)
            self.assertEqual(provenance["clinical_indicator_count"], 21)
            schema = json.loads((root / "result/scalar_features.schema.json").read_text())
            self.assertEqual(
                schema["missing_value_policy"],
                "preserve_nan; fit median imputation on training only",
            )
            self.assertTrue((root / "result/dcm/0001.npz").is_file())


if __name__ == "__main__":
    unittest.main()

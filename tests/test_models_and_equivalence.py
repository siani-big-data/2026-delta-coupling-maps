import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from baroreflex_pipeline.analysis.paired import analyze_prediction_file, paired_equivalence
from baroreflex_pipeline.models import (
    ClinicalReferenceModel,
    DCMReferenceModel,
    ScalarReferenceModel,
    trainable_parameter_count,
)


class ModelAndEquivalenceTests(unittest.TestCase):
    def test_reference_parameter_counts(self):
        self.assertEqual(trainable_parameter_count(ScalarReferenceModel(3880)), 515617)
        self.assertEqual(trainable_parameter_count(ClinicalReferenceModel(21)), 13409)
        self.assertEqual(trainable_parameter_count(DCMReferenceModel((25, 25))), 167217)

    def test_paired_equivalence_decision(self):
        predictions = pd.DataFrame(
            {
                "record_id": [f"{index:04d}" for index in range(20)],
                "target": [30.0] * 20,
                "scalar_prediction": [30.0] * 20,
                "dcm_prediction": [30.1] * 20,
            }
        )
        result = paired_equivalence(predictions, margin_years=0.5, confidence=0.90)
        self.assertTrue(result["equivalent"])
        self.assertAlmostEqual(result["comparator_mae_years"], 0.0)
        self.assertAlmostEqual(result["dcm_mae_years"], 0.1)

    def test_clinical_panel_can_be_primary_comparator(self):
        predictions = pd.DataFrame(
            {
                "record_id": [f"{index:04d}" for index in range(20)],
                "target": [30.0] * 20,
                "clinical_prediction": [30.0] * 20,
                "dcm_prediction": [30.1] * 20,
            }
        )
        result = paired_equivalence(
            predictions,
            margin_years=0.5,
            confidence=0.90,
            comparator_column="clinical_prediction",
        )
        self.assertEqual(result["comparator_column"], "clinical_prediction")
        self.assertTrue(result["equivalent"])

    def test_nonfinite_predictions_are_rejected_from_equivalence_analysis(self):
        predictions = pd.DataFrame(
            {
                "record_id": ["0001", "0002"],
                "split": ["test", "test"],
                "target": [30.0, 40.0],
                "clinical_prediction": [35.0, 35.0],
                "dcm_prediction": [35.0, float("inf")],
            }
        )
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "predictions.csv"
            predictions.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "must be finite"):
                analyze_prediction_file(
                    source,
                    Path(temporary) / "equivalence.json",
                    margin_years=0.5,
                    confidence=0.90,
                    comparator_column="clinical_prediction",
                )


if __name__ == "__main__":
    unittest.main()

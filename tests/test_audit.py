import unittest

import pandas as pd

from baroreflex_pipeline.config import PipelineConfig
from baroreflex_pipeline.processing.audit import rr_morphology_metrics
from baroreflex_pipeline.processing.signals import beat_quality_summary
from tests.support import PROCESSING_CONFIG


class AuditTests(unittest.TestCase):
    def test_beat_quality_summary_counts_each_rule(self) -> None:
        beats = pd.DataFrame(
            {
                "rr_s": [0.8, 0.2, 0.9],
                "sbp_mmhg": [120.0, 120.0, 65.0],
                "dbp_mmhg": [80.0, 80.0, 90.0],
                "mbp_mmhg": [93.3, 93.3, 81.7],
                "valid": [True, False, False],
            }
        )

        summary = beat_quality_summary(beats, PipelineConfig.from_json(PROCESSING_CONFIG))

        self.assertEqual(summary["total_beats"], 3)
        self.assertEqual(summary["valid_beats"], 1)
        self.assertEqual(summary["rr_rejected"], 1)
        self.assertEqual(summary["sbp_rejected"], 1)
        self.assertEqual(summary["dbp_rejected"], 0)
        self.assertEqual(summary["mbp_rejected"], 0)
        self.assertEqual(summary["pressure_order_rejected"], 1)

    def test_rr_morphology_uses_only_adjacent_accepted_intervals(self) -> None:
        beats = pd.DataFrame(
            {
                "beat_index": [0, 1, 2, 3],
                "rr_s": [0.8, 0.81, 1.5, 0.82],
                "valid": [True, True, False, True],
            }
        )

        metrics = rr_morphology_metrics(beats)

        self.assertEqual(metrics["accepted_rr"], 3)
        self.assertAlmostEqual(metrics["rmssd_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()

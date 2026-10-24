import unittest

import numpy as np
import pandas as pd

from baroreflex_pipeline.representations.clinical import (
    CLINICAL_COLUMNS,
    extract_clinical_indicators,
    extract_clinical_panel,
)


def synthetic_beats() -> pd.DataFrame:
    sbp = np.r_[100, 102, 104, 106, 104, 102, 100, 102, 104, 106].astype(float)
    rr_ms = 5.0 * sbp + 300.0
    return pd.DataFrame(
        {
            "beat_index": np.arange(len(sbp)),
            "rr_s": rr_ms / 1000.0,
            "r_amplitude": np.ones(len(sbp)),
            "sbp_mmhg": sbp,
            "dbp_mmhg": sbp - 40.0,
            "mbp_mmhg": sbp - 26.7,
            "valid": np.ones(len(sbp), dtype=bool),
        }
    )


class ClinicalIndicatorTests(unittest.TestCase):
    def test_indicator_panel_has_fixed_clinical_schema(self):
        panel = extract_clinical_panel({"1": synthetic_beats()})
        self.assertEqual(panel.index.tolist(), ["0001"])
        self.assertEqual(tuple(panel.columns), CLINICAL_COLUMNS)
        self.assertEqual(panel.shape, (1, 21))

    def test_sequence_method_recovers_known_slope(self):
        indicators = extract_clinical_indicators(synthetic_beats())
        self.assertAlmostEqual(indicators["brs_up_ms_per_mmhg"], 5.0)
        self.assertAlmostEqual(indicators["brs_down_ms_per_mmhg"], 5.0)
        self.assertGreater(indicators["bei_all"], 0.0)


if __name__ == "__main__":
    unittest.main()

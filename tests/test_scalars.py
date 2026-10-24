import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from baroreflex_pipeline.representations.scalars import extract_scalar_panel, prepare_scalar_series


def beat_table(offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(80)
    return pd.DataFrame(
        {
            "beat_index": index,
            "rr_s": 0.8 + offset + 0.02 * np.sin(index / 3),
            "r_amplitude": 1.0 + 0.1 * np.cos(index / 4),
            "sbp_mmhg": 115 + 3 * np.sin(index / 5),
            "dbp_mmhg": 70 + 2 * np.sin(index / 5),
            "mbp_mmhg": 85 + 2.3 * np.sin(index / 5),
            "valid": np.r_[np.ones(79, dtype=bool), False],
        }
    )


class ScalarTests(unittest.TestCase):
    def test_undefined_raw_features_are_not_imputed_before_split(self):
        extracted = pd.DataFrame(
            {"undefined": [np.nan, np.inf], "observed": [1, 9999]}, index=["0001", "0002"]
        )
        with patch(
            "baroreflex_pipeline.representations.scalars.extract_features", return_value=extracted
        ):
            panel = extract_scalar_panel({"0001": beat_table(), "0002": beat_table()})
        self.assertTrue(panel["undefined"].isna().all())
        self.assertEqual(panel["observed"].tolist(), [1, 9999])

    def test_preparation_excludes_invalid_beats_and_covariates(self):
        prepared = prepare_scalar_series("0001", beat_table())
        self.assertEqual(len(prepared), 79)
        self.assertEqual(
            list(prepared.columns),
            ["record_id", "beat_index", "rr_s", "r_amplitude", "sbp_mmhg", "dbp_mmhg", "mbp_mmhg"],
        )

    def test_minimal_panel_has_one_row_per_record(self):
        panel = extract_scalar_panel(
            {"0001": beat_table(), "0002": beat_table(0.01)},
            feature_set="minimal",
            n_jobs=1,
        )
        self.assertEqual(panel.index.tolist(), ["0001", "0002"])
        self.assertGreater(panel.shape[1], 20)
        self.assertFalse(panel.isna().any().any())


if __name__ == "__main__":
    unittest.main()

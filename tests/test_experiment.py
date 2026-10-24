import unittest

import pandas as pd

from baroreflex_pipeline.data.cohort import create_subject_splits, load_experiment_config
from tests.support import EXPERIMENT_CONFIG


class ExperimentTests(unittest.TestCase):
    def test_fixed_stratified_subject_split(self):
        rows = []
        for age_group in range(1, 16):
            for repetition in range(20):
                record = age_group * 100 + repetition
                rows.append(
                    {
                        "record_id": f"{record:04d}",
                        "status": "ok",
                        "Age_group": age_group,
                        "Sex": repetition % 2,
                        "BMI": 20 + repetition / 10,
                        "Length": 15,
                        "Device": repetition % 2,
                    }
                )
        manifest = pd.DataFrame(rows)
        config = load_experiment_config(EXPERIMENT_CONFIG)
        first, audit = create_subject_splits(manifest, config)
        second, _ = create_subject_splits(manifest, config)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(audit["eligible_count"], 300)
        self.assertEqual(audit["split_counts"], {"train": 210, "validation": 45, "test": 45})
        self.assertTrue(first.groupby("record_id").size().eq(1).all())
        self.assertTrue(first["age_midpoint_years"].notna().all())


if __name__ == "__main__":
    unittest.main()

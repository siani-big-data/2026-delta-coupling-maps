import unittest

import numpy as np
import pandas as pd

from baroreflex_pipeline.config import PipelineConfig
from baroreflex_pipeline.representations.dcm import (
    build_dcm,
    compose_dcm_rgb,
    consecutive_deltas,
    estimate_response_lag,
)


def config() -> PipelineConfig:
    return PipelineConfig(
        dataset_url="https://example.invalid/",
        trim_start_seconds=0,
        minimum_valid_beats=3,
        ecg_channel_names=("ECG",),
        bp_channel_names=("NIBP",),
        rr_seconds_range=(0.3, 2.0),
        sbp_mmhg_range=(70, 200),
        dbp_mmhg_range=(40, 120),
        mbp_mmhg_range=(50, 140),
        dcm_bins=25,
        delta_rr_ms_range=(-200, 200),
        delta_sbp_mmhg_range=(-10, 10),
        maximum_lag_beats=6,
    )


class DCMTests(unittest.TestCase):
    def test_lag_estimation_detects_delayed_rr_response(self):
        rng = np.random.default_rng(42)
        delta_sbp = rng.normal(size=400)
        delta_rr = np.concatenate((np.zeros(3), delta_sbp[:-3])) + rng.normal(scale=0.02, size=400)
        lag, correlations = estimate_response_lag(delta_sbp, delta_rr, max_lag=6)
        self.assertEqual(lag, 3)
        self.assertGreater(correlations[3], 0.99)

    def test_zero_lag_is_valid(self):
        signal = np.linspace(-1, 1, 20)
        lag, _ = estimate_response_lag(signal, signal, max_lag=4)
        self.assertEqual(lag, 0)

    def test_deltas_do_not_bridge_invalid_beats(self):
        beats = pd.DataFrame(
            {
                "rr_s": [0.8, 0.9, 1.0, 1.1],
                "sbp_mmhg": [110, 111, 112, 113],
                "valid": [True, False, True, True],
            }
        )
        delta_sbp, delta_rr = consecutive_deltas(beats)
        np.testing.assert_array_equal(delta_sbp, [1.0])
        np.testing.assert_allclose(delta_rr, [100.0])

    def test_dcm_preserves_both_channels(self):
        n = 80
        beats = pd.DataFrame(
            {
                "rr_s": 0.8 + 0.02 * np.sin(np.arange(n) / 3),
                "sbp_mmhg": 110 + 2 * np.sin(np.arange(n) / 3),
                "valid": np.ones(n, dtype=bool),
            }
        )
        result = build_dcm(beats, config())
        self.assertEqual(result.channels.shape, (2, 25, 25))
        self.assertTrue(np.any(result.channels[0] > 0))
        self.assertTrue(np.any(result.channels[1] > 0))
        self.assertGreaterEqual(result.estimated_lag_beats, 0)

    def test_rgb_composition_uses_blue_orange_and_neutral_overlap(self):
        channels = np.zeros((2, 1, 3), dtype=float)
        channels[0, 0, 0] = 1.0
        channels[1, 0, 1] = 1.0
        channels[:, 0, 2] = 1.0

        rgb = compose_dcm_rgb(channels)

        np.testing.assert_allclose(rgb[0, 0], [0.0, 0.447, 0.698])
        np.testing.assert_allclose(rgb[0, 1], [0.835, 0.369, 0.0])
        np.testing.assert_allclose(rgb[0, 2], [0.88, 0.88, 0.88])


if __name__ == "__main__":
    unittest.main()

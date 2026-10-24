from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired_equivalence(
    predictions: pd.DataFrame,
    margin_years: float,
    confidence: float,
    comparator_column: str = "scalar_prediction",
) -> dict:
    required = {"record_id", "target", comparator_column, "dcm_prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if predictions["record_id"].duplicated().any():
        raise ValueError("Predictions must contain exactly one row per test subject")
    values = predictions[list(required - {"record_id"})].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions and targets must be finite")

    comparator_error = np.abs(predictions[comparator_column] - predictions["target"])
    dcm_error = np.abs(predictions["dcm_prediction"] - predictions["target"])
    paired_difference = dcm_error - comparator_error
    mean_difference = float(paired_difference.mean())
    standard_error = float(stats.sem(paired_difference))
    if len(predictions) < 2 or not np.isfinite(standard_error) or np.isclose(standard_error, 0.0):
        lower = upper = mean_difference
    else:
        lower, upper = stats.t.interval(
            confidence,
            df=len(predictions) - 1,
            loc=mean_difference,
            scale=standard_error,
        )
    return {
        "participant_count": int(len(predictions)),
        "comparator_column": comparator_column,
        "comparator_mae_years": float(comparator_error.mean()),
        "dcm_mae_years": float(dcm_error.mean()),
        "paired_difference_dcm_minus_comparator_years": mean_difference,
        "confidence_level": confidence,
        "confidence_interval_years": [float(lower), float(upper)],
        "equivalence_margin_years": margin_years,
        "equivalent": bool(lower > -margin_years and upper < margin_years),
    }


def analyze_prediction_file(
    predictions_path: Path,
    output_path: Path,
    margin_years: float,
    confidence: float,
    comparator_column: str = "scalar_prediction",
) -> dict:
    predictions = pd.read_csv(predictions_path, dtype={"record_id": str})
    if "split" in predictions.columns:
        predictions = predictions.loc[predictions["split"].eq("test")].copy()
    result = paired_equivalence(
        predictions, margin_years, confidence, comparator_column=comparator_column
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

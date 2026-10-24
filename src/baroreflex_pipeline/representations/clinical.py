from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CLINICAL_COLUMNS = (
    "mean_rr_ms",
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50_pct",
    "mean_sbp_mmhg",
    "sd_sbp_mmhg",
    "mean_dbp_mmhg",
    "sd_dbp_mmhg",
    "mean_mbp_mmhg",
    "sd_mbp_mmhg",
    "brs_up_ms_per_mmhg",
    "brs_down_ms_per_mmhg",
    "brs_all_ms_per_mmhg",
    "bei_up",
    "bei_down",
    "bei_all",
    "baroreflex_up_sequences",
    "baroreflex_down_sequences",
    "sbp_up_ramps",
    "sbp_down_ramps",
)


def _true_runs(mask: np.ndarray, minimum_steps: int = 2) -> list[tuple[int, int]]:

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum_steps:
                runs.append((start, index))
            start = None
    return runs


def _sequence_slopes(
    sbp: np.ndarray,
    rr_ms: np.ndarray,
    valid: np.ndarray,
    direction: int,
    lag_beats: int = 1,
    sbp_threshold_mmhg: float = 1.0,
    rr_threshold_ms: float = 5.0,
    minimum_beats: int = 3,
    minimum_r_squared: float = 0.85,
) -> tuple[list[float], int]:
    if lag_beats < 0:
        raise ValueError("lag_beats must be non-negative")
    if lag_beats == 0:
        x, y, paired_valid = sbp, rr_ms, valid.copy()
    else:
        x, y = sbp[:-lag_beats], rr_ms[lag_beats:]
        paired_valid = valid[:-lag_beats] & valid[lag_beats:]

    dx = np.diff(x)
    dy = np.diff(y)
    contiguous = paired_valid[:-1] & paired_valid[1:]
    if direction > 0:
        sequence_steps = contiguous & (dx >= sbp_threshold_mmhg) & (dy >= rr_threshold_ms)
        ramp_steps = (np.diff(sbp) >= sbp_threshold_mmhg) & valid[:-1] & valid[1:]
    else:
        sequence_steps = contiguous & (dx <= -sbp_threshold_mmhg) & (dy <= -rr_threshold_ms)
        ramp_steps = (np.diff(sbp) <= -sbp_threshold_mmhg) & valid[:-1] & valid[1:]

    sequence_runs = _true_runs(sequence_steps, minimum_beats - 1)
    ramp_count = len(_true_runs(ramp_steps, minimum_beats - 1))
    slopes: list[float] = []
    for start, stop in sequence_runs:
        sequence_x = x[start : stop + 1]
        sequence_y = y[start : stop + 1]
        correlation = np.corrcoef(sequence_x, sequence_y)[0, 1]
        if np.isfinite(correlation) and correlation**2 >= minimum_r_squared:
            slopes.append(float(np.polyfit(sequence_x, sequence_y, 1)[0]))
    return slopes, ramp_count


def extract_clinical_indicators(beats: pd.DataFrame) -> dict[str, float]:

    required = {"rr_s", "sbp_mmhg", "dbp_mmhg", "mbp_mmhg", "valid"}
    missing = required.difference(beats.columns)
    if missing:
        raise ValueError(f"Beat table is missing columns: {sorted(missing)}")

    valid = beats["valid"].to_numpy(dtype=bool)
    rr_ms = beats["rr_s"].to_numpy(dtype=float) * 1000.0
    sbp = beats["sbp_mmhg"].to_numpy(dtype=float)
    dbp = beats["dbp_mmhg"].to_numpy(dtype=float)
    mbp = beats["mbp_mmhg"].to_numpy(dtype=float)
    if valid.sum() < 3:
        raise ValueError("Fewer than three valid beats remain")

    adjacent = valid[:-1] & valid[1:]
    rr_differences = np.diff(rr_ms)[adjacent]
    accepted_rr = rr_ms[valid]
    up_slopes, up_ramps = _sequence_slopes(sbp, rr_ms, valid, direction=1)
    down_slopes, down_ramps = _sequence_slopes(sbp, rr_ms, valid, direction=-1)
    all_slopes = up_slopes + down_slopes

    def mean_or_nan(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    return {
        "mean_rr_ms": float(np.mean(accepted_rr)),
        "mean_hr_bpm": float(np.mean(60000.0 / accepted_rr)),
        "sdnn_ms": float(np.std(accepted_rr, ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(rr_differences**2))),
        "pnn50_pct": float(100.0 * np.mean(np.abs(rr_differences) > 50.0)),
        "mean_sbp_mmhg": float(np.mean(sbp[valid])),
        "sd_sbp_mmhg": float(np.std(sbp[valid], ddof=1)),
        "mean_dbp_mmhg": float(np.mean(dbp[valid])),
        "sd_dbp_mmhg": float(np.std(dbp[valid], ddof=1)),
        "mean_mbp_mmhg": float(np.mean(mbp[valid])),
        "sd_mbp_mmhg": float(np.std(mbp[valid], ddof=1)),
        "brs_up_ms_per_mmhg": mean_or_nan(up_slopes),
        "brs_down_ms_per_mmhg": mean_or_nan(down_slopes),
        "brs_all_ms_per_mmhg": mean_or_nan(all_slopes),
        "bei_up": ratio(len(up_slopes), up_ramps),
        "bei_down": ratio(len(down_slopes), down_ramps),
        "bei_all": ratio(len(all_slopes), up_ramps + down_ramps),
        "baroreflex_up_sequences": float(len(up_slopes)),
        "baroreflex_down_sequences": float(len(down_slopes)),
        "sbp_up_ramps": float(up_ramps),
        "sbp_down_ramps": float(down_ramps),
    }


def extract_clinical_panel(beat_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {
        str(record_id).zfill(4): extract_clinical_indicators(beats)
        for record_id, beats in beat_tables.items()
    }
    panel = pd.DataFrame.from_dict(rows, orient="index", columns=CLINICAL_COLUMNS)
    panel.index.name = "record_id"
    return panel.sort_index()


def save_clinical_panel(panel: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=True)
    schema = {
        "panel": "clinical_indicators_v1",
        "feature_count": len(CLINICAL_COLUMNS),
        "sequence_method": {
            "lag_beats": 1,
            "minimum_beats": 3,
            "sbp_step_threshold_mmhg": 1.0,
            "rr_step_threshold_ms": 5.0,
            "minimum_r_squared": 0.85,
            "bei_definition": "accepted_baroreflex_sequences_divided_by_direction_matched_sbp_ramps",
        },
        "missing_value_policy": "Preserve undefined sequence indices as NaN; impute from training data only.",
        "contains_target_or_demographics": False,
        "columns": list(CLINICAL_COLUMNS),
    }
    output_path.with_suffix(".schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )


def extract_clinical_panel_from_directory(beats_dir: Path, output_path: Path) -> pd.DataFrame:
    paths = sorted(beats_dir.glob("*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No beat files found in {beats_dir}")
    beat_tables = {path.name.removesuffix(".csv.gz"): pd.read_csv(path) for path in paths}
    panel = extract_clinical_panel(beat_tables)
    save_clinical_panel(panel, output_path)
    return panel

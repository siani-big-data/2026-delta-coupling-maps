from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baroreflex_pipeline.config import PipelineConfig
from baroreflex_pipeline.processing.signals import beat_quality_summary, extract_beat_series

EXCLUSION_AUDIT_COLUMNS = (
    "record_id",
    "audit_status",
    "reason",
    "total_beats",
    "valid_beats",
    "valid_fraction",
    "rr_rejected",
    "sbp_rejected",
    "dbp_rejected",
    "mbp_rejected",
    "pressure_order_rejected",
    "dominant_rejection",
)


def audit_excluded_records(
    raw_dir: Path,
    manifest_path: Path,
    output_path: Path,
    config: PipelineConfig,
) -> pd.DataFrame:

    manifest = pd.read_csv(manifest_path, dtype={"record_id": str})
    excluded = manifest.loc[manifest["status"].ne("ok"), "record_id"].str.zfill(4)
    rows: list[dict[str, object]] = []

    for record_id in excluded:
        row: dict[str, object] = {
            "record_id": record_id,
            "audit_status": "unreadable",
            "reason": "",
        }
        try:
            record = extract_beat_series(
                raw_dir / record_id,
                config,
                enforce_minimum_beats=False,
            )
            summary = beat_quality_summary(record.beats, config)
            rejection_counts = {
                name: int(summary[f"{name}_rejected"])
                for name in ("rr", "sbp", "dbp", "mbp", "pressure_order")
            }
            row.update(summary)
            row["dominant_rejection"] = max(rejection_counts, key=rejection_counts.get)
            row["audit_status"] = (
                "eligible"
                if summary["valid_beats"] >= config.minimum_valid_beats
                else "insufficient_valid_beats"
            )
        except Exception as error:
            row["reason"] = f"{type(error).__name__}: {error}"
        rows.append(row)

    audit = pd.DataFrame(rows).reindex(columns=EXCLUSION_AUDIT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    return audit


def rr_morphology_metrics(beats: pd.DataFrame) -> dict[str, float | int]:

    accepted = beats.loc[beats["valid"], ["beat_index", "rr_s"]].copy()
    rr_ms = accepted["rr_s"].to_numpy(dtype=float) * 1000.0
    adjacent = np.diff(accepted["beat_index"].to_numpy(dtype=int)) == 1
    rr_differences = np.diff(rr_ms)[adjacent]
    if rr_ms.size < 2 or rr_differences.size == 0:
        raise ValueError("At least two adjacent accepted RR intervals are required")
    return {
        "accepted_rr": int(rr_ms.size),
        "sdnn_ms": float(np.std(rr_ms, ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(rr_differences**2))),
        "small_delta_5ms_pct": float(100.0 * np.mean(np.abs(rr_differences) <= 5.0)),
        "unique_rr_fraction": float(np.unique(rr_ms).size / rr_ms.size),
    }


def audit_rr_morphology(
    beats_dir: Path,
    cohort_path: Path,
    output_path: Path,
    lower_quantile: float = 0.01,
) -> pd.DataFrame:

    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must lie between zero and 0.5")
    cohort = pd.read_csv(cohort_path, dtype={"record_id": str})
    cohort["record_id"] = cohort["record_id"].str.zfill(4)
    rows: list[dict[str, object]] = []
    for path in sorted(beats_dir.glob("*.csv.gz")):
        record_id = path.name.split(".", maxsplit=1)[0].zfill(4)
        metrics = rr_morphology_metrics(pd.read_csv(path))
        rows.append({"record_id": record_id, **metrics})

    if not rows:
        raise ValueError(f"No beat files were found in {beats_dir}")
    audit = pd.DataFrame(rows).merge(
        cohort[["record_id", "split"]], on="record_id", how="left", validate="one_to_one"
    )
    training = audit.loc[audit["split"].eq("train")]
    if training.empty:
        raise ValueError("The cohort file contains no training records with beat files")
    upper_quantile = 1.0 - lower_quantile
    sdnn_low = float(training["sdnn_ms"].quantile(lower_quantile))
    rmssd_low = float(training["rmssd_ms"].quantile(lower_quantile))
    sdnn_high = float(training["sdnn_ms"].quantile(upper_quantile))
    rmssd_high = float(training["rmssd_ms"].quantile(upper_quantile))
    audit["near_constant_rr"] = (audit["sdnn_ms"] < sdnn_low) & (audit["rmssd_ms"] < rmssd_low)
    audit["extreme_rr_variability"] = (audit["sdnn_ms"] > sdnn_high) | (
        audit["rmssd_ms"] > rmssd_high
    )
    audit["training_sdnn_lower_ms"] = sdnn_low
    audit["training_rmssd_lower_ms"] = rmssd_low
    audit["training_sdnn_upper_ms"] = sdnn_high
    audit["training_rmssd_upper_ms"] = rmssd_high
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    return audit

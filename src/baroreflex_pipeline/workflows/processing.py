from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from baroreflex_pipeline.config import PipelineConfig
from baroreflex_pipeline.processing.signals import extract_beat_series, save_beat_series
from baroreflex_pipeline.representations.base import TabularFeatureExtractor
from baroreflex_pipeline.representations.dcm import build_dcm, plot_dcm, save_dcm
from baroreflex_pipeline.representations.extractors import ClinicalExtractor, TsfreshExtractor


def _load_metadata(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "subject-info.csv"
    if not path.exists():
        return pd.DataFrame()
    metadata = pd.read_csv(path, dtype={"ID": str})
    metadata["ID"] = metadata["ID"].str.zfill(4)
    return metadata.set_index("ID", drop=False)


def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    records: list[str] | None = None,
    save_beats: bool = True,
    generate_scalars: bool = True,
    generate_clinical: bool = True,
    scalar_feature_set: str = "efficient",
    scalar_n_jobs: int = 0,
    save_figures: bool = True,
) -> pd.DataFrame:

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata(raw_dir)
    available = sorted(path.stem for path in raw_dir.glob("*.hea"))
    requested = [record.zfill(4) for record in records] if records else available
    manifest_rows: list[dict[str, object]] = []
    beat_tables: dict[str, pd.DataFrame] = {}

    for record_id in requested:
        row: dict[str, object] = {"record_id": record_id, "status": "failed", "reason": ""}
        try:
            record = extract_beat_series(raw_dir / record_id, config)
            dcm = build_dcm(record.beats, config)
            if generate_scalars or generate_clinical:
                beat_tables[record_id] = record.beats
            if save_beats:
                save_beat_series(record, output_dir / "beats" / f"{record_id}.csv.gz")
            save_dcm(dcm, output_dir / "dcm" / f"{record_id}.npz")
            if save_figures:
                plot_dcm(
                    dcm,
                    output_dir / "figures" / f"{record_id}.png",
                    title=f"DCM - subject {record_id}",
                )
            row.update(
                {
                    "status": "ok",
                    "sampling_rate_hz": record.sampling_rate_hz,
                    "ecg_channel": record.ecg_channel,
                    "bp_channel": record.bp_channel,
                    "total_beats": len(record.beats),
                    "valid_beats": int(record.beats["valid"].sum()),
                    "valid_transitions": dcm.valid_transition_count,
                    "estimated_lag_beats": dcm.estimated_lag_beats,
                }
            )
            if record_id in metadata.index:
                for field in ("Age_group", "Sex", "BMI", "Length", "Device"):
                    row[field] = metadata.loc[record_id, field]
        except Exception as error:
            row["reason"] = f"{type(error).__name__}: {error}"
        manifest_rows.append(row)
        print(f"[{row['status'].upper()}] {record_id} {row['reason']}")

    extractors: list[tuple[TabularFeatureExtractor, str]] = []
    feature_counts = {}
    if generate_scalars:
        extractors.append(
            (TsfreshExtractor(scalar_feature_set, scalar_n_jobs), "scalar_features.csv")
        )
    if generate_clinical:
        extractors.append((ClinicalExtractor(), "clinical_indicators.csv"))
    if beat_tables:
        for extractor, filename in extractors:
            print(f"[FEATURES] {type(extractor).__name__}: {len(beat_tables)} records", flush=True)
            features = extractor.extract(beat_tables)
            feature_counts[filename] = int(features.shape[1])
            extractor.save(features, output_dir / filename)
            print(f"[FEATURES COMPLETE] {filename}: shape={features.shape}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    provenance = {
        "dataset": "Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging",
        "dataset_version": "1.0.0",
        "dataset_doi": "10.13026/2hsy-t491",
        "configuration": config.__dict__,
        "requested_records": requested,
        "scalar_feature_set": scalar_feature_set if generate_scalars else None,
        "scalar_feature_count": feature_counts.get("scalar_features.csv", 0),
        "clinical_indicator_count": feature_counts.get("clinical_indicators.csv", 0),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=list), encoding="utf-8"
    )
    return manifest

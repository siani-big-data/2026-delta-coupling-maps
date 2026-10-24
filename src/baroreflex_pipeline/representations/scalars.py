from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)

SCALAR_SERIES = ("rr_s", "r_amplitude", "sbp_mmhg", "dbp_mmhg", "mbp_mmhg")


def _feature_parameters(feature_set: str) -> dict:
    settings = {
        "minimal": MinimalFCParameters,
        "efficient": EfficientFCParameters,
        "comprehensive": ComprehensiveFCParameters,
    }
    try:
        parameters = settings[feature_set]()
    except KeyError as error:
        raise ValueError(f"Unknown scalar feature set: {feature_set}") from error

    parameters.pop("query_similarity_count", None)
    return parameters


def prepare_scalar_series(record_id: str, beats: pd.DataFrame) -> pd.DataFrame:

    required = {"beat_index", "valid", *SCALAR_SERIES}
    missing = required.difference(beats.columns)
    if missing:
        raise ValueError(f"Beat table is missing columns: {sorted(missing)}")
    accepted = beats.loc[beats["valid"], ["beat_index", *SCALAR_SERIES]].copy()
    accepted.insert(0, "record_id", record_id)
    accepted.replace([np.inf, -np.inf], np.nan, inplace=True)
    accepted.dropna(subset=list(SCALAR_SERIES), inplace=True)
    if len(accepted) < 3:
        raise ValueError(f"Record {record_id} has fewer than three usable beats")
    return accepted


def extract_scalar_panel(
    beat_tables: dict[str, pd.DataFrame],
    feature_set: str = "efficient",
    n_jobs: int = 0,
) -> pd.DataFrame:

    if not beat_tables:
        return pd.DataFrame(index=pd.Index([], name="record_id"))
    long_table = pd.concat(
        [prepare_scalar_series(record_id, beats) for record_id, beats in beat_tables.items()],
        ignore_index=True,
    )
    features = extract_features(
        long_table,
        column_id="record_id",
        column_sort="beat_index",
        default_fc_parameters=_feature_parameters(feature_set),
        disable_progressbar=False,
        n_jobs=n_jobs,
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.index = features.index.astype(str).str.zfill(4)
    features.index.name = "record_id"
    return features.sort_index().sort_index(axis=1)


def save_scalar_panel(
    features: pd.DataFrame,
    output_path: Path,
    feature_set: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=True)
    schema = {
        "feature_set": feature_set,
        "source_series": list(SCALAR_SERIES),
        "record_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "contains_target_or_demographics": False,
        "missing_value_policy": "preserve_nan; fit median imputation on training only",
        "columns": features.columns.tolist(),
    }
    output_path.with_suffix(".schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )


def extract_scalar_panel_from_directory(
    beats_dir: Path,
    output_path: Path,
    feature_set: str = "efficient",
    n_jobs: int = 0,
) -> pd.DataFrame:
    paths = sorted(beats_dir.glob("*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No beat files found in {beats_dir}")
    beat_tables = {path.name.removesuffix(".csv.gz"): pd.read_csv(path) for path in paths}
    features = extract_scalar_panel(beat_tables, feature_set=feature_set, n_jobs=n_jobs)
    save_scalar_panel(features, output_path, feature_set)
    return features

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from baroreflex_pipeline.config import load_experiment_config


def create_subject_splits(
    manifest: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:

    covariates = config["covariates"]["columns"]
    required = ["record_id", "status", config["target"]["source"], *covariates]
    missing = set(required).difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing experiment columns: {sorted(missing)}")

    eligible = manifest.loc[manifest["status"].eq("ok"), required].copy()
    complete = eligible[required[2:]].notna().all(axis=1)
    eligible = eligible.loc[complete].copy()
    eligible["record_id"] = eligible["record_id"].astype(str).str.zfill(4)
    eligible["Age_group"] = eligible["Age_group"].astype(int)
    midpoints = {int(key): value for key, value in config["target"]["age_group_midpoints"].items()}
    eligible[config["target"]["name"]] = eligible[config["target"]["source"]].map(midpoints)

    split = config["split"]
    train, temporary = train_test_split(
        eligible,
        test_size=split["validation_fraction"] + split["test_fraction"],
        random_state=split["seed"],
        stratify=eligible[split["stratify_by"]],
    )
    relative_test = split["test_fraction"] / (split["validation_fraction"] + split["test_fraction"])
    validation, test = train_test_split(
        temporary,
        test_size=relative_test,
        random_state=split["seed"],
        stratify=temporary[split["stratify_by"]],
    )
    assignments = []
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        part = frame.copy()
        part["split"] = name
        assignments.append(part)
    output = pd.concat(assignments, ignore_index=True).sort_values("record_id")

    scaling: dict[str, dict[str, float]] = {}
    for column in config["covariates"]["continuous_columns"]:
        mean = float(train[column].mean())
        std = float(train[column].std(ddof=0))
        scaling[column] = {"mean": mean, "standard_deviation": std}
    audit = {
        "manifest_ok_count": int(manifest["status"].eq("ok").sum()),
        "excluded_missing_target_or_covariate_count": int((~complete).sum()),
        "eligible_count": int(len(output)),
        "split_counts": output["split"].value_counts().to_dict(),
        "continuous_covariate_training_statistics": scaling,
        "configuration": config,
    }
    return output, audit


def save_subject_splits(manifest_path: Path, output_dir: Path, config_path: Path) -> pd.DataFrame:
    config = load_experiment_config(config_path)
    manifest = pd.read_csv(manifest_path, dtype={"record_id": str})
    splits, audit = create_subject_splits(manifest, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output_dir / "splits.csv", index=False)
    (output_dir / "experiment_protocol.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return splits

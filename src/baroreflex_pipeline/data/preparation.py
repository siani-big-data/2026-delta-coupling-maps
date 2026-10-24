from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _standardize_from_training(
    values: np.ndarray,
    train_mask: np.ndarray,
    *,
    impute_missing: bool,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    values = np.asarray(values, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    if values.ndim != 2 or train_mask.shape != (len(values),) or not train_mask.any():
        raise ValueError("Expected a matrix and a nonempty aligned training mask")
    values = np.where(np.isfinite(values), values, np.nan)
    training = values[train_mask]
    observed = np.isfinite(training).any(axis=0)
    medians = np.zeros(values.shape[1])
    medians[observed] = np.nanmedian(training[:, observed], axis=0)
    if impute_missing:
        values = np.where(np.isfinite(values), values, medians)
        training = values[train_mask]
    elif not np.isfinite(values).all():
        raise ValueError(
            "Non-finite values remain in a representation that does not permit imputation"
        )
    means = np.mean(training, axis=0)
    means[~np.isfinite(means)] = 0.0
    standard_deviations = np.std(training, axis=0, ddof=0)
    constant = ~np.isfinite(standard_deviations) | (standard_deviations == 0)
    safe_standard_deviations = standard_deviations.copy()
    safe_standard_deviations[constant] = 1.0
    standardized = (values - means) / safe_standard_deviations
    standardized[:, constant] = 0.0
    return standardized.astype(np.float32), {
        "training_medians": medians.tolist(),
        "training_means": means.tolist(),
        "training_standard_deviations": standard_deviations.tolist(),
        "constant_training_columns": np.flatnonzero(constant).astype(int).tolist(),
    }


def _shared_arrays(splits: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    splits = splits.copy().sort_values("record_id").reset_index(drop=True)
    train_mask = splits["split"].eq("train").to_numpy()
    covariate_columns = ["Sex", "BMI", "Length", "Device"]
    covariates = splits[covariate_columns].to_numpy(dtype=float)
    continuous_indices = [covariate_columns.index("BMI"), covariate_columns.index("Length")]
    continuous, scaling = _standardize_from_training(
        covariates[:, continuous_indices], train_mask, impute_missing=False
    )
    covariates[:, continuous_indices] = continuous
    shared = {
        "record_id": splits["record_id"].astype(str).str.zfill(4).to_numpy(dtype="U4"),
        "split": splits["split"].to_numpy(dtype="U10"),
        "target": splits["age_midpoint_years"].to_numpy(dtype=np.float32),
        "covariates": covariates.astype(np.float32),
    }
    metadata = {
        "covariate_columns": covariate_columns,
        "continuous_covariate_columns": ["BMI", "Length"],
        "continuous_scaling": scaling,
    }
    return shared, metadata


def _load_tabular_representation(
    path: Path,
    record_ids: np.ndarray,
    train_mask: np.ndarray,
    *,
    impute_missing: bool,
) -> tuple[np.ndarray, list[str], dict[str, list[float]]]:
    table = pd.read_csv(path, dtype={"record_id": str}).set_index("record_id")
    table.index = table.index.astype(str).str.zfill(4)
    missing = sorted(set(record_ids).difference(table.index))
    if missing:
        raise ValueError(
            f"{path.name} is missing {len(missing)} cohort records; first: {missing[:5]}"
        )
    table = table.loc[record_ids]
    columns = table.columns.tolist()
    values, scaling = _standardize_from_training(
        table.to_numpy(dtype=float), train_mask, impute_missing=impute_missing
    )
    return values, columns, scaling


def _load_dcms(dcm_dir: Path, record_ids: np.ndarray) -> np.ndarray:
    arrays: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for record_id in record_ids:
        path = dcm_dir / f"{record_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing DCM for cohort record {record_id}: {path}")
        with np.load(path) as artifact:
            dcm = np.asarray(artifact["dcm"], dtype=np.float32)
        expected_shape = expected_shape or dcm.shape
        if dcm.shape != expected_shape:
            raise ValueError(
                f"Inconsistent DCM shape for {record_id}: {dcm.shape} != {expected_shape}"
            )
        arrays.append(dcm)
    return np.stack(arrays)


def prepare_training_artifacts(
    output_dir: Path,
    training_dir: Path | None = None,
) -> dict[str, object]:

    training_dir = training_dir or output_dir / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(output_dir / "splits.csv", dtype={"record_id": str})
    splits["record_id"] = splits["record_id"].str.zfill(4)
    if splits["record_id"].duplicated().any():
        raise ValueError("Duplicate participant IDs in split assignments")
    if set(splits["split"]) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test partitions")
    schema_path = output_dir / "scalar_features.schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
        if (
            schema.get("missing_value_policy")
            != "preserve_nan; fit median imputation on training only"
        ):
            raise ValueError(
                "Regenerate scalar features: their train-only imputation provenance is unverified"
            )
    shared, shared_metadata = _shared_arrays(splits)
    train_mask = shared["split"] == "train"
    record_ids = shared["record_id"]

    specifications = {
        "clinical": {
            "path": output_dir / "clinical_indicators.csv",
            "impute_missing": True,
        },
        "scalar": {
            "path": output_dir / "scalar_features.csv",
            "impute_missing": True,
        },
    }
    metadata: dict[str, object] = {
        "preprocessing_revision": "train_only_median_v1",
        "record_count": int(record_ids.size),
        "split_counts": pd.Series(shared["split"]).value_counts().to_dict(),
        "target": "age_midpoint_years",
        **shared_metadata,
        "representations": {},
    }

    for name, specification in specifications.items():
        inputs, columns, scaling = _load_tabular_representation(
            specification["path"],
            record_ids,
            train_mask,
            impute_missing=bool(specification["impute_missing"]),
        )
        np.savez_compressed(training_dir / f"{name}.npz", inputs=inputs, **shared)
        metadata["representations"][name] = {
            "input_shape": list(inputs.shape[1:]),
            "feature_count": int(inputs.shape[1]),
            "feature_columns": columns,
            "training_only_preprocessing": scaling,
        }
        print(f"[PREPARE] {name}: inputs={inputs.shape}", flush=True)

    dcm_inputs = _load_dcms(output_dir / "dcm", record_ids)
    np.savez_compressed(training_dir / "dcm.npz", inputs=dcm_inputs, **shared)
    metadata["representations"]["dcm"] = {
        "input_shape": list(dcm_inputs.shape[1:]),
        "feature_count": int(np.prod(dcm_inputs.shape[1:])),
        "training_only_preprocessing": "none; subject-level joint maximum normalization is stored in each DCM",
    }
    print(f"[PREPARE] dcm: inputs={dcm_inputs.shape}", flush=True)
    (training_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    splits.to_csv(training_dir / "subjects.csv", index=False)
    return metadata

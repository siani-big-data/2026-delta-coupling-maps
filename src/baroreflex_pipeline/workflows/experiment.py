from __future__ import annotations

import json
from pathlib import Path

from baroreflex_pipeline.config import PipelineConfig
from baroreflex_pipeline.data.cohort import save_subject_splits
from baroreflex_pipeline.data.download import download_dataset
from baroreflex_pipeline.data.preparation import prepare_training_artifacts
from baroreflex_pipeline.processing.audit import audit_excluded_records, audit_rr_morphology
from baroreflex_pipeline.training.backends import TorchTrainer
from baroreflex_pipeline.workflows.processing import process_dataset


def run_complete_workflow(
    raw_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    *,
    experiment_config: Path,
    records: list[str] | None = None,
    download: bool = True,
    save_figures: bool = False,
    scalar_feature_set: str = "efficient",
    scalar_n_jobs: int = 0,
    training_seeds: list[int] | None = None,
    training_device: str = "auto",
    training_models: list[str] | None = None,
    training_epochs: int | None = None,
    training_patience: int | None = None,
    training_batch_size: int | None = None,
    training_learning_rate: float | None = None,
    training_weight_decay: float | None = None,
) -> dict[str, object]:

    trainer = TorchTrainer()
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, str]] = []

    def start(name: str) -> None:
        print(f"[STAGE START] {name}", flush=True)

    def complete(name: str, artifact: Path) -> None:
        stages.append({"stage": name, "status": "completed", "artifact": str(artifact)})
        (output_dir / "workflow_status.json").write_text(
            json.dumps({"trainer": trainer.name, "stages": stages}, indent=2), encoding="utf-8"
        )
        print(f"[STAGE COMPLETE] {name} -> {artifact}", flush=True)

    if download:
        start("download")
        download_dataset(config.dataset_url, raw_dir)
        complete("download", raw_dir)

    start("process_signals_and_representations")
    process_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        config=config,
        records=records,
        save_beats=True,
        generate_scalars=True,
        generate_clinical=True,
        scalar_feature_set=scalar_feature_set,
        scalar_n_jobs=scalar_n_jobs,
        save_figures=save_figures,
    )
    complete("process_signals_and_representations", output_dir / "manifest.csv")

    start("subject_split")
    save_subject_splits(output_dir / "manifest.csv", output_dir, experiment_config)
    complete("subject_split", output_dir / "splits.csv")

    start("excluded_signal_audit")
    audit_excluded_records(
        raw_dir, output_dir / "manifest.csv", output_dir / "excluded_signal_audit.csv", config
    )
    complete("excluded_signal_audit", output_dir / "excluded_signal_audit.csv")
    start("rr_morphology_audit")
    audit_rr_morphology(
        output_dir / "beats",
        output_dir / "splits.csv",
        output_dir / "rr_morphology_audit.csv",
    )
    complete("rr_morphology_audit", output_dir / "rr_morphology_audit.csv")

    start("prepare_training_data")
    prepare_training_artifacts(output_dir)
    complete("prepare_training_data", output_dir / "training_data" / "metadata.json")

    start(f"{trainer.name}_training")
    trainer.fit(
        output_dir / "training_data",
        output_dir / "runs" / trainer.name,
        experiment_config=experiment_config,
        seeds=training_seeds,
        device=training_device,
        models=training_models,
        epochs_maximum=training_epochs,
        patience=training_patience,
        batch_size=training_batch_size,
        learning_rate=training_learning_rate,
        weight_decay=training_weight_decay,
    )
    complete(f"{trainer.name}_training", output_dir / "runs" / trainer.name / "summary.json")
    return {"trainer": trainer.name, "stages": stages}

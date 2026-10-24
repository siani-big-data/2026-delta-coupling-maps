from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import PipelineConfig
from ..models.specs import MODEL_INPUTS
from .parser import _parser


class CliApplication:
    def __init__(self, arguments: argparse.Namespace, parser: argparse.ArgumentParser):
        self.arguments = arguments
        self.parser = parser

    def execute(self) -> None:
        command = self.arguments.command.replace("-", "_")
        getattr(self, f"_handle_{command}")()

    def _processing_config(self) -> PipelineConfig:
        if self.arguments.config is None:
            self.parser.error("This command requires --config PATH before the subcommand")
        return PipelineConfig.from_json(self.arguments.config)

    def _handle_models(self) -> None:
        for name, representation in MODEL_INPUTS.items():
            print(f"{name:38s} input={representation}")

    def _handle_status(self) -> None:
        directory = self.arguments.run_dir
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for filename in ("run_manifest.json", "ablation_provenance.json", "rerun_provenance.json"):
            path = directory / filename
            if path.exists():
                print(f"status={json.loads(path.read_text()).get('status', 'unknown')}")
                break
        required = ("run.json", "best.pt", "predictions.csv", "history.csv")
        for folder in sorted(path for path in directory.iterdir() if path.is_dir()):
            completed = [
                path
                for path in folder.glob("seed_*")
                if all((path / filename).is_file() for filename in required)
            ]
            if completed:
                print(f"{folder.name}: {len(completed)} completed seeds")

    def _handle_predict(self) -> None:
        from ..training.checkpoints import predict_checkpoint

        args = self.arguments
        predict_checkpoint(
            args.checkpoint,
            args.inputs,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(args.output)

    def _handle_scalars(self) -> None:
        from ..representations.extractors import TsfreshExtractor

        args = self.arguments
        extractor = TsfreshExtractor(args.feature_set, args.jobs)
        extractor.extract_from_directory(args.beats_dir, args.output_file)

    def _handle_clinical(self) -> None:
        from ..representations.extractors import ClinicalExtractor

        args = self.arguments
        ClinicalExtractor().extract_from_directory(args.beats_dir, args.output_file)

    def _handle_audit_exclusions(self) -> None:
        from ..processing.audit import audit_excluded_records

        args = self.arguments
        report = audit_excluded_records(
            args.raw_dir, args.manifest, args.output, self._processing_config()
        )
        print(report["audit_status"].value_counts(dropna=False).to_string())

    def _handle_audit_rr(self) -> None:
        from ..processing.audit import audit_rr_morphology

        args = self.arguments
        report = audit_rr_morphology(args.beats_dir, args.cohort, args.output, args.tail_quantile)
        print(report[["near_constant_rr", "extreme_rr_variability"]].sum().to_string())

    def _handle_split(self) -> None:
        from ..data.cohort import save_subject_splits

        args = self.arguments
        save_subject_splits(args.manifest, args.output_dir, args.experiment_config)

    def _handle_prepare_training(self) -> None:
        from ..data.preparation import prepare_training_artifacts

        args = self.arguments
        prepare_training_artifacts(args.output_dir, args.training_dir)

    def _handle_train(self) -> None:
        from ..training.backends import TorchTrainer

        args = self.arguments
        trainer = TorchTrainer()
        run_dir = args.run_dir or Path("outputs/experiments/default/runs") / trainer.name
        trainer.fit(
            args.training_dir,
            run_dir,
            experiment_config=args.experiment_config,
            seeds=args.seeds,
            epochs_maximum=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
            models=args.models,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    def _handle_equivalence(self) -> None:
        from ..analysis.paired import analyze_prediction_file

        args = self.arguments
        analyze_prediction_file(
            args.predictions,
            args.output,
            margin_years=args.margin_years,
            confidence=args.confidence,
            comparator_column=args.comparator_column,
        )

    def _handle_representatives(self) -> None:
        from ..analysis.representatives import plot_representative_dcms, select_representative_dcms

        args = self.arguments
        selected = select_representative_dcms(
            args.dcm_dir, args.cohort, tuple(args.age_groups), minimum_lag=args.minimum_lag
        )
        plot_representative_dcms(selected, args.output)
        for item in selected:
            print(
                f"[SELECTED] group={item.age_group} record={item.record_id} "
                f"lag={item.lag_beats} distance={item.distance_to_subgroup_median:.6f}"
            )

    def _handle_workflow(self) -> None:
        from ..workflows.experiment import run_complete_workflow

        args = self.arguments
        run_complete_workflow(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            config=self._processing_config(),
            experiment_config=args.experiment_config,
            records=args.records,
            download=not args.skip_download,
            save_figures=args.figures,
            scalar_feature_set=args.scalar_feature_set,
            scalar_n_jobs=args.scalar_jobs,
            training_seeds=args.training_seeds,
            training_device=args.training_device,
            training_models=args.models,
            training_epochs=args.epochs,
            training_patience=args.patience,
            training_batch_size=args.batch_size,
            training_learning_rate=args.learning_rate,
            training_weight_decay=args.weight_decay,
        )

    def _handle_download(self) -> None:
        from ..data.download import download_dataset

        download_dataset(self._processing_config().dataset_url, self.arguments.raw_dir)

    def _handle_process(self) -> None:
        from ..workflows.processing import process_dataset

        args = self.arguments
        process_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            config=self._processing_config(),
            records=args.records,
            save_beats=not args.no_beat_files,
            generate_scalars=not args.no_scalars,
            generate_clinical=not args.no_clinical,
            scalar_feature_set=args.scalar_feature_set,
            scalar_n_jobs=args.scalar_jobs,
            save_figures=not args.no_figures,
        )

    def _handle_run(self) -> None:
        self._handle_download()
        self._handle_process()


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    CliApplication(parser.parse_args(argv), parser).execute()

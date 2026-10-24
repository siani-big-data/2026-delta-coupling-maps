import argparse
from pathlib import Path

from ..models.specs import MODEL_INPUTS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build scalar and two-dimensional baroreflex representations from the PhysioNet Autonomic Aging dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit processing JSON, required for commands that process signals.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download", help="Download the public PhysioNet dataset with resumable transfers."
    )
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    process = commands.add_parser(
        "process", help="Extract beat series and generate scalar, clinical, and DCM products."
    )
    process.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    process.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/default"))
    process.add_argument("--records", nargs="*", help="Optional record IDs, for example: 0001 0002")
    process.add_argument("--no-beat-files", action="store_true")
    process.add_argument("--no-figures", action="store_true")
    process.add_argument("--no-scalars", action="store_true")
    process.add_argument("--no-clinical", action="store_true")
    process.add_argument(
        "--scalar-feature-set",
        choices=("minimal", "efficient", "comprehensive"),
        default="efficient",
    )
    process.add_argument(
        "--scalar-jobs", type=int, default=0, help="tsfresh worker count; 0 uses its default."
    )

    run = commands.add_parser("run", help="Download and process the dataset.")
    run.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    run.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/default"))
    run.add_argument("--records", nargs="*")
    run.add_argument("--no-beat-files", action="store_true")
    run.add_argument("--no-figures", action="store_true")
    run.add_argument("--no-scalars", action="store_true")
    run.add_argument("--no-clinical", action="store_true")
    run.add_argument(
        "--scalar-feature-set",
        choices=("minimal", "efficient", "comprehensive"),
        default="efficient",
    )
    run.add_argument("--scalar-jobs", type=int, default=0)

    workflow = commands.add_parser(
        "workflow",
        help="Run download, signal analysis, feature/DCM generation, splitting, and training.",
    )
    workflow.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    workflow.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/default"))
    workflow.add_argument("--experiment-config", type=Path, required=True)
    workflow.add_argument("--records", nargs="*")
    workflow.add_argument("--skip-download", action="store_true")
    workflow.add_argument("--figures", action="store_true")
    workflow.add_argument(
        "--scalar-feature-set",
        choices=("minimal", "efficient", "comprehensive"),
        default="efficient",
    )
    workflow.add_argument("--scalar-jobs", type=int, default=0)
    workflow.add_argument("--training-seeds", nargs="+", type=int)
    workflow.add_argument(
        "--training-device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    workflow.add_argument("--models", nargs="+", choices=tuple(MODEL_INPUTS))
    workflow.add_argument("--epochs", type=int)
    workflow.add_argument("--patience", type=int)
    workflow.add_argument("--batch-size", type=int)
    workflow.add_argument("--learning-rate", type=float)
    workflow.add_argument("--weight-decay", type=float)

    scalars = commands.add_parser(
        "scalars", help="Generate the scalar panel from existing beat files."
    )
    scalars.add_argument(
        "--beats-dir", type=Path, default=Path("outputs/experiments/default/beats")
    )
    scalars.add_argument(
        "--output-file", type=Path, default=Path("outputs/experiments/default/scalar_features.csv")
    )
    scalars.add_argument(
        "--feature-set", choices=("minimal", "efficient", "comprehensive"), default="efficient"
    )
    scalars.add_argument("--jobs", type=int, default=0)

    clinical = commands.add_parser(
        "clinical", help="Generate the clinically interpretable indicator panel from beat files."
    )
    clinical.add_argument(
        "--beats-dir", type=Path, default=Path("outputs/experiments/default/beats")
    )
    clinical.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/experiments/default/clinical_indicators.csv"),
    )

    audit = commands.add_parser(
        "audit-exclusions",
        help="Explain failed records by counting each beat-level quality rejection.",
    )
    audit.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    audit.add_argument(
        "--manifest", type=Path, default=Path("outputs/experiments/default/manifest.csv")
    )
    audit.add_argument(
        "--output", type=Path, default=Path("outputs/experiments/default/excluded_signal_audit.csv")
    )

    rr_audit = commands.add_parser(
        "audit-rr",
        help="Flag near-constant and extremely variable RR morphology using training-only thresholds.",
    )
    rr_audit.add_argument(
        "--beats-dir", type=Path, default=Path("outputs/experiments/default/beats")
    )
    rr_audit.add_argument(
        "--cohort", type=Path, default=Path("outputs/experiments/default/splits.csv")
    )
    rr_audit.add_argument(
        "--output", type=Path, default=Path("outputs/experiments/default/rr_morphology_audit.csv")
    )
    rr_audit.add_argument("--tail-quantile", type=float, default=0.01)

    split = commands.add_parser(
        "split", help="Create the fixed subject-level experiment partitions."
    )
    split.add_argument(
        "--manifest", type=Path, default=Path("outputs/experiments/default/manifest.csv")
    )
    split.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/default"))
    split.add_argument("--experiment-config", type=Path, required=True)

    prepare = commands.add_parser(
        "prepare-training", help="Build aligned model-ready NPZ artifacts."
    )
    prepare.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/default"))
    prepare.add_argument("--training-dir", type=Path)

    train = commands.add_parser("train", help="Train PyTorch models over prepared artifacts.")
    train.add_argument(
        "--training-dir", type=Path, default=Path("outputs/experiments/default/training_data")
    )
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--experiment-config", type=Path, required=True)
    train.add_argument("--seeds", nargs="+", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--patience", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    train.add_argument("--models", nargs="+", choices=tuple(MODEL_INPUTS))
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)

    commands.add_parser("models", help="List supported input model architectures.")
    predict = commands.add_parser(
        "predict", help="Generate predictions from a checkpoint and prepared NPZ inputs."
    )
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--inputs", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    predict.add_argument("--batch-size", type=int, default=64)
    status = commands.add_parser(
        "status", help="Show completed seeds and status without starting a run."
    )
    status.add_argument("--run-dir", type=Path, required=True)

    equivalence = commands.add_parser(
        "equivalence", help="Evaluate paired scalar-versus-DCM predictions."
    )
    equivalence.add_argument("--predictions", type=Path, required=True)
    equivalence.add_argument(
        "--output", type=Path, default=Path("outputs/experiments/default/equivalence.json")
    )
    equivalence.add_argument("--margin-years", type=float, default=0.5)
    equivalence.add_argument("--confidence", type=float, default=0.90)
    equivalence.add_argument(
        "--comparator-column",
        default="clinical_prediction",
        help="Prediction column paired against dcm_prediction.",
    )

    representatives = commands.add_parser(
        "representatives", help="Select age-group medoids and render a publication figure."
    )
    representatives.add_argument(
        "--dcm-dir", type=Path, default=Path("outputs/experiments/default/dcm")
    )
    representatives.add_argument(
        "--cohort", type=Path, default=Path("outputs/experiments/default/splits.csv")
    )
    representatives.add_argument(
        "--output", type=Path, default=Path("outputs/experiments/default/representative_dcms.png")
    )
    representatives.add_argument("--age-groups", nargs="+", type=int, default=[2, 9, 12])
    representatives.add_argument("--minimum-lag", type=int, default=1)
    return parser

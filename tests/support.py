from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = REPOSITORY / "configs/experiment.json"
PROCESSING_CONFIG = REPOSITORY / "configs/default.json"

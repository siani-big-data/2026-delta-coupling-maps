import json
from pathlib import Path

import numpy as np


def load_experiment_config(path: Path) -> dict:
    if path is None:
        raise ValueError("An experiment configuration file is required")
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    fractions = config["split"]
    values = [fractions[key] for key in ("train_fraction", "validation_fraction", "test_fraction")]
    if not np.isfinite(values).all() or min(values) <= 0 or not np.isclose(sum(values), 1.0):
        raise ValueError("Positive train, validation, and test fractions must sum to one")
    return config

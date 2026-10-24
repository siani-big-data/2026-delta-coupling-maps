from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    dataset_url: str
    trim_start_seconds: float
    minimum_valid_beats: int
    ecg_channel_names: tuple[str, ...]
    bp_channel_names: tuple[str, ...]
    rr_seconds_range: tuple[float, float]
    sbp_mmhg_range: tuple[float, float]
    dbp_mmhg_range: tuple[float, float]
    mbp_mmhg_range: tuple[float, float]
    dcm_bins: int
    delta_rr_ms_range: tuple[float, float]
    delta_sbp_mmhg_range: tuple[float, float]
    maximum_lag_beats: int

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(values)

    @classmethod
    def from_dict(cls, values: dict) -> "PipelineConfig":
        values = dict(values)
        for key in (
            "ecg_channel_names",
            "bp_channel_names",
            "rr_seconds_range",
            "sbp_mmhg_range",
            "dbp_mmhg_range",
            "mbp_mmhg_range",
            "delta_rr_ms_range",
            "delta_sbp_mmhg_range",
        ):
            values[key] = tuple(values[key])
        return cls(**values)

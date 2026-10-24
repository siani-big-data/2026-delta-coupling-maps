from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

from baroreflex_pipeline.config import PipelineConfig

BEAT_COLUMNS = ["beat_index", "rr_s", "r_amplitude", "sbp_mmhg", "dbp_mmhg", "mbp_mmhg", "valid"]


@dataclass(frozen=True)
class RecordResult:
    record_id: str
    beats: pd.DataFrame
    sampling_rate_hz: float
    ecg_channel: str
    bp_channel: str


def _find_channel(names: list[str], candidates: tuple[str, ...]) -> int:
    normalized = {name.strip().upper(): index for index, name in enumerate(names)}
    for candidate in candidates:
        if candidate.upper() in normalized:
            return normalized[candidate.upper()]
    raise ValueError(f"None of the required channels {candidates} were found in {names}")


def _interpolate_finite(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal)
    if finite.sum() < 2:
        raise ValueError("Signal contains fewer than two finite samples")
    if finite.all():
        return signal
    locations = np.arange(signal.size)
    return np.interp(locations, locations[finite], signal[finite])


def beat_quality_summary(beats: pd.DataFrame, config: PipelineConfig) -> dict[str, int | float]:

    checks = {
        "rr": beats["rr_s"].between(*config.rr_seconds_range),
        "sbp": beats["sbp_mmhg"].between(*config.sbp_mmhg_range),
        "dbp": beats["dbp_mmhg"].between(*config.dbp_mmhg_range),
        "mbp": beats["mbp_mmhg"].between(*config.mbp_mmhg_range),
        "pressure_order": beats["sbp_mmhg"] > beats["dbp_mmhg"],
    }
    summary: dict[str, int | float] = {
        "total_beats": len(beats),
        "valid_beats": int(beats["valid"].sum()),
        "valid_fraction": float(beats["valid"].mean()) if len(beats) else 0.0,
    }
    summary.update(
        {f"{name}_rejected": int((~accepted).sum()) for name, accepted in checks.items()}
    )
    return summary


def extract_beat_series(
    record_path: Path,
    config: PipelineConfig,
    *,
    enforce_minimum_beats: bool = True,
) -> RecordResult:

    record_path = record_path.with_suffix("")
    record = wfdb.rdrecord(str(record_path))
    fs = float(record.fs)
    start = int(round(config.trim_start_seconds * fs))
    if record.p_signal.shape[0] <= start:
        raise ValueError("Recording is shorter than the configured initial trim")

    ecg_index = _find_channel(record.sig_name, config.ecg_channel_names)
    bp_index = _find_channel(record.sig_name, config.bp_channel_names)
    ecg = _interpolate_finite(record.p_signal[start:, ecg_index])
    bp = _interpolate_finite(record.p_signal[start:, bp_index])

    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    peaks = np.asarray(peak_info["ECG_R_Peaks"], dtype=int)
    if peaks.size < 3:
        raise ValueError("Fewer than three ECG R peaks were detected")

    rows: list[dict[str, float | int | bool]] = []
    for beat_index, (left, right) in enumerate(zip(peaks[:-1], peaks[1:])):
        segment = bp[left:right]
        if segment.size == 0:
            continue
        rr_s = (right - left) / fs
        sbp = float(np.max(segment))
        dbp = float(np.min(segment))
        mbp = dbp + (sbp - dbp) / 3.0
        valid = (
            config.rr_seconds_range[0] <= rr_s <= config.rr_seconds_range[1]
            and config.sbp_mmhg_range[0] <= sbp <= config.sbp_mmhg_range[1]
            and config.dbp_mmhg_range[0] <= dbp <= config.dbp_mmhg_range[1]
            and config.mbp_mmhg_range[0] <= mbp <= config.mbp_mmhg_range[1]
            and sbp > dbp
        )
        rows.append(
            {
                "beat_index": beat_index,
                "rr_s": rr_s,
                "r_amplitude": float(ecg[right]),
                "sbp_mmhg": sbp,
                "dbp_mmhg": dbp,
                "mbp_mmhg": mbp,
                "valid": valid,
            }
        )

    beats = pd.DataFrame(rows, columns=BEAT_COLUMNS)
    if enforce_minimum_beats and int(beats["valid"].sum()) < config.minimum_valid_beats:
        raise ValueError("Fewer than the configured number of valid beats remain")
    return RecordResult(
        record_id=record_path.name,
        beats=beats,
        sampling_rate_hz=fs,
        ecg_channel=record.sig_name[ecg_index].strip(),
        bp_channel=record.sig_name[bp_index].strip(),
    )


def save_beat_series(result: RecordResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.beats.to_csv(output_path, index=False, compression="gzip")

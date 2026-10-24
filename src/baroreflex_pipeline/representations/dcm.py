from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baroreflex_pipeline.config import PipelineConfig


@dataclass(frozen=True)
class DCMResult:
    channels: np.ndarray
    immediate_counts: np.ndarray
    delayed_counts: np.ndarray
    rr_edges_ms: np.ndarray
    sbp_edges_mmhg: np.ndarray
    estimated_lag_beats: int
    lag_correlations: np.ndarray
    valid_transition_count: int


def consecutive_deltas(beats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:

    valid = beats["valid"].to_numpy(dtype=bool)
    adjacent_valid = valid[:-1] & valid[1:]
    rr_ms = beats["rr_s"].to_numpy(dtype=float) * 1000.0
    sbp = beats["sbp_mmhg"].to_numpy(dtype=float)
    return np.diff(sbp)[adjacent_valid], np.diff(rr_ms)[adjacent_valid]


def estimate_response_lag(
    delta_sbp: np.ndarray, delta_rr: np.ndarray, max_lag: int
) -> tuple[int, np.ndarray]:

    if delta_sbp.size != delta_rr.size:
        raise ValueError("Delta arrays must have equal length")
    correlations = np.full(max_lag + 1, np.nan, dtype=float)
    for lag in range(max_lag + 1):
        x = delta_sbp if lag == 0 else delta_sbp[:-lag]
        y = delta_rr if lag == 0 else delta_rr[lag:]
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
            continue
        correlations[lag] = np.corrcoef(x[finite], y[finite])[0, 1]
    if np.all(np.isnan(correlations)):
        return 0, correlations
    return int(np.nanargmax(correlations)), correlations


def _histogram(
    delta_sbp: np.ndarray, delta_rr: np.ndarray, sbp_edges: np.ndarray, rr_edges: np.ndarray
) -> np.ndarray:
    counts, _, _ = np.histogram2d(delta_sbp, delta_rr, bins=[sbp_edges, rr_edges])
    return counts.T


def build_dcm(beats: pd.DataFrame, config: PipelineConfig) -> DCMResult:
    delta_sbp, delta_rr = consecutive_deltas(beats)
    if delta_sbp.size < 3:
        raise ValueError("Fewer than three valid beat-to-beat transitions remain")

    lag, correlations = estimate_response_lag(delta_sbp, delta_rr, config.maximum_lag_beats)
    sbp_edges = np.linspace(*config.delta_sbp_mmhg_range, config.dcm_bins + 1)
    rr_edges = np.linspace(*config.delta_rr_ms_range, config.dcm_bins + 1)
    immediate = _histogram(delta_sbp, delta_rr, sbp_edges, rr_edges)

    if lag == 0:
        delayed = immediate.copy()
    else:
        delayed = _histogram(delta_sbp[:-lag], delta_rr[lag:], sbp_edges, rr_edges)

    scale = max(float(immediate.max()), float(delayed.max()), 1.0)
    channels = np.stack((immediate / scale, delayed / scale)).astype(np.float32)
    return DCMResult(
        channels=channels,
        immediate_counts=immediate.astype(np.int32),
        delayed_counts=delayed.astype(np.int32),
        rr_edges_ms=rr_edges,
        sbp_edges_mmhg=sbp_edges,
        estimated_lag_beats=lag,
        lag_correlations=correlations,
        valid_transition_count=int(delta_sbp.size),
    )


def save_dcm(result: DCMResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        dcm=result.channels,
        immediate_counts=result.immediate_counts,
        delayed_counts=result.delayed_counts,
        rr_edges_ms=result.rr_edges_ms,
        sbp_edges_mmhg=result.sbp_edges_mmhg,
        estimated_lag_beats=np.array(result.estimated_lag_beats),
        lag_correlations=result.lag_correlations,
        valid_transition_count=np.array(result.valid_transition_count),
    )


def compose_dcm_rgb(channels: np.ndarray) -> np.ndarray:

    channels = np.asarray(channels, dtype=float)
    if channels.ndim != 3 or channels.shape[0] != 2:
        raise ValueError("DCM channels must have shape (2, height, width)")
    immediate, delayed = np.clip(channels, 0.0, 1.0)
    total = immediate + delayed
    balance = np.divide(
        immediate - delayed,
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    immediate_color = np.array([0.0, 0.447, 0.698])
    delayed_color = np.array([0.835, 0.369, 0.0])
    overlap_color = np.array([0.88, 0.88, 0.88])

    dominance = np.sqrt(np.abs(balance))[..., None]
    endpoint = np.where((balance >= 0)[..., None], immediate_color, delayed_color)
    hue = overlap_color * (1.0 - dominance) + endpoint * dominance

    brightness = np.sqrt(np.maximum(immediate, delayed))
    return brightness[..., None] * hue


def plot_dcm(result: DCMResult, output_path: Path, title: str | None = None) -> None:

    rgb = compose_dcm_rgb(result.channels)
    extent = [
        result.sbp_edges_mmhg[0],
        result.sbp_edges_mmhg[-1],
        result.rr_edges_ms[0],
        result.rr_edges_ms[-1],
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    ax.imshow(rgb, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"$\Delta$SBP (mmHg)")
    ax.set_ylabel(r"$\Delta$RR (ms)")
    ax.set_title(title or f"DCM (estimated lag: {result.estimated_lag_beats} beats)")
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)

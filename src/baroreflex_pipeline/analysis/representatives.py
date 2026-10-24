from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from baroreflex_pipeline.representations.dcm import compose_dcm_rgb


@dataclass(frozen=True)
class RepresentativeDCM:
    record_id: str
    age_group: int
    age_label: str
    lag_beats: int
    valid_transitions: int
    distance_to_subgroup_median: float
    path: Path


def _age_label(age_midpoint: float) -> str:
    midpoint = int(round(age_midpoint))
    return f"{midpoint - 2}-{midpoint + 2} years"


def select_representative_dcms(
    dcm_dir: Path,
    cohort_path: Path,
    age_groups: tuple[int, ...],
    minimum_lag: int = 1,
) -> list[RepresentativeDCM]:

    cohort = pd.read_csv(cohort_path, dtype={"record_id": str})
    cohort["record_id"] = cohort["record_id"].str.zfill(4)
    selected: list[RepresentativeDCM] = []

    for age_group in age_groups:
        candidates: list[dict[str, object]] = []
        group = cohort[(cohort["status"] == "ok") & (cohort["Age_group"] == age_group)]
        for row in group.itertuples(index=False):
            path = dcm_dir / f"{row.record_id}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                lag = int(data["estimated_lag_beats"])
                if lag < minimum_lag:
                    continue
                candidates.append(
                    {
                        "record_id": row.record_id,
                        "age_midpoint": float(row.age_midpoint_years),
                        "path": path,
                        "channels": data["dcm"].copy(),
                        "lag": lag,
                        "valid_transitions": int(data["valid_transition_count"]),
                    }
                )
        if not candidates:
            raise ValueError(f"No DCMs found for age group {age_group} with lag >= {minimum_lag}")

        stack = np.stack([item["channels"] for item in candidates])
        median = np.median(stack, axis=0)
        distances = np.sqrt(np.mean((stack - median) ** 2, axis=(1, 2, 3)))
        ranked = sorted(
            zip(candidates, distances),
            key=lambda pair: (
                float(pair[1]),
                -int(pair[0]["valid_transitions"]),
                str(pair[0]["record_id"]),
            ),
        )
        item, distance = ranked[0]
        selected.append(
            RepresentativeDCM(
                record_id=str(item["record_id"]),
                age_group=age_group,
                age_label=_age_label(float(item["age_midpoint"])),
                lag_beats=int(item["lag"]),
                valid_transitions=int(item["valid_transitions"]),
                distance_to_subgroup_median=float(distance),
                path=Path(item["path"]),
            )
        )
    return selected


def plot_representative_dcms(representatives: list[RepresentativeDCM], output_path: Path) -> None:

    fig, axes = plt.subplots(
        1, len(representatives), figsize=(11.0, 3.45), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, representative in zip(axes, representatives):
        with np.load(representative.path) as data:
            rgb = compose_dcm_rgb(data["dcm"])
            sbp_edges = data["sbp_edges_mmhg"]
            rr_edges = data["rr_edges_ms"]
        extent = [sbp_edges[0], sbp_edges[-1], rr_edges[0], rr_edges[-1]]
        ax.imshow(rgb, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.55)
        ax.axvline(0, color="white", linewidth=0.5, alpha=0.55)
        plural = "s" if representative.lag_beats != 1 else ""
        ax.set_title(
            f"{representative.age_label}\nID {representative.record_id}; "
            f"lag {representative.lag_beats} beat{plural}",
            fontsize=10,
        )
        ax.set_xlabel(r"$\Delta$SBP (mmHg)")
    axes[0].set_ylabel(r"$\Delta$RR (ms)")
    legend = [
        Patch(facecolor="#0072B2", label="Immediate dominant"),
        Patch(facecolor="#D55E00", label="Delayed dominant"),
        Patch(facecolor="#E0E0E0", label="Equal overlap"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.83, bottom=0.25, wspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "record_id": item.record_id,
                "age_group": item.age_group,
                "age_label": item.age_label,
                "estimated_lag_beats": item.lag_beats,
                "valid_transitions": item.valid_transitions,
                "distance_to_subgroup_median": item.distance_to_subgroup_median,
            }
            for item in representatives
        ]
    ).to_csv(output_path.with_suffix(".csv"), index=False)

#!/usr/bin/env python3
"""Render Fe-B-Cr mixing enthalpy with embedded phase boundaries snapped to the enthalpy grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from enthalpy_config import DEFAULT_DATABASE_PATH, TERNARY_STEP
from enthalpy_core import (
    build_fraction_vectors,
    compute_multi_component_enthalpy,
    load_omega_tables,
    normalize_step,
)
from enthalpy_plot import build_ternary_figure, write_plotly_image


# -------------------------- Grid + enthalpy -------------------------- #
def compute_fe_b_cr(tables, step: float):
    """Generate ternary grid and enthalpy values for Fe-B-Cr."""
    total_units, actual_step = normalize_step(step)
    vectors = build_fraction_vectors(3, total_units)
    if not vectors:
        raise RuntimeError("No ternary compositions produced with the provided step.")

    fe_pct: List[float] = []
    b_pct: List[float] = []
    cr_pct: List[float] = []
    enthalpies: List[float] = []

    for fe_units, cr_units, b_units in vectors:
        fe = fe_units / total_units
        cr = cr_units / total_units
        b = b_units / total_units
        delta_h, _ = compute_multi_component_enthalpy(tables, [("Fe", fe), ("Cr", cr), ("B", b)])
        fe_pct.append(fe * 100)
        cr_pct.append(cr * 100)
        b_pct.append(b * 100)
        enthalpies.append(delta_h)

    return fe_pct, b_pct, cr_pct, enthalpies, actual_step


# -------------------------- Boundary cleaning -------------------------- #
def _resolve_dataset_columns(df: pd.DataFrame) -> None:
    """Forward-fill dataset names so each a/b/c column tuple is labeled."""
    resolved_names = []
    current_name: str | None = None
    for dataset_name, _axis in df.columns:
        dataset_str = str(dataset_name)
        if not dataset_str.startswith("Unnamed"):
            current_name = dataset_str.strip()
        if not current_name:
            raise ValueError("Encountered unlabeled dataset column before any named columns.")
        resolved_names.append(current_name)

    axes = [str(axis).strip().lower() for axis in df.columns.get_level_values(1)]
    df.columns = pd.MultiIndex.from_arrays([resolved_names, axes], names=["dataset", "axis"])


def _drop_duplicate_rows(coords: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if len(coords) <= 1:
        return coords
    filtered = [coords[0]]
    for row in coords[1:]:
        if np.linalg.norm(row - filtered[-1]) > tol:
            filtered.append(row)
    return np.array(filtered)


def _monotonic_order(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 2:
        return coords
    ranges = np.ptp(coords, axis=0)
    primary_axis = int(np.argmax(ranges))
    order = np.argsort(coords[:, primary_axis], kind="mergesort")
    return coords[order]


def _snap_to_grid(coords: np.ndarray, step: float) -> np.ndarray:
    """Snap barycentric rows to the nearest grid defined by step (fraction units)."""
    if step <= 0:
        return coords
    snapped = np.clip(coords, 0.0, 1.0)
    snapped = np.round(snapped / step) * step
    sums = snapped.sum(axis=1, keepdims=True)
    valid = sums.squeeze() > 0
    snapped[valid] /= sums[valid]
    return snapped


def load_boundaries_snapped(csv_path: Path, grid_step: float) -> Dict[str, pd.DataFrame]:
    """Load phase boundaries, clean, sort monotonically, and snap to ternary grid."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, header=[0, 1])
    _resolve_dataset_columns(df)

    ordered_names = list(dict.fromkeys(df.columns.get_level_values("dataset")))
    boundaries: Dict[str, pd.DataFrame] = {}

    for name in ordered_names:
        subset = df[name]
        if subset.empty:
            continue
        cleaned = subset.dropna(how="all")
        cleaned = cleaned.dropna(subset=["a", "b", "c"], how="any")
        if cleaned.empty:
            continue
        numeric = cleaned.apply(pd.to_numeric, errors="coerce").dropna(subset=["a", "b", "c"])
        if numeric.empty:
            continue

        coords = numeric[["a", "b", "c"]].to_numpy(dtype=float)
        coords = coords[np.isfinite(coords).all(axis=1)]
        if len(coords) == 0:
            continue
        coords = np.clip(coords, 0.0, 1.0)
        sums = coords.sum(axis=1, keepdims=True)
        valid = sums.squeeze() > 0
        coords[valid] /= sums[valid]
        coords = _monotonic_order(coords)
        coords = _snap_to_grid(coords, grid_step)
        coords = _drop_duplicate_rows(coords)
        boundaries[name] = pd.DataFrame(coords, columns=["a", "b", "c"])

    return boundaries


# -------------------------- Plot helpers -------------------------- #
def _edge_labels(num_labels: Iterable[int]) -> list[go.Scatterternary]:
    texts = [str(v) for v in num_labels]
    font = dict(color="black", size=12)

    fb = go.Scatterternary(
        a=[v for v in num_labels],  # B%
        b=[100 - v for v in num_labels],  # Fe%
        c=[0 for _ in num_labels],
        mode="markers+text",
        marker=dict(size=2, color="rgba(0,0,0,0)"),
        text=texts,
        textposition="middle left",
        textfont=font,
        hoverinfo="skip",
        showlegend=False,
        cliponaxis=False,
    )

    fc = go.Scatterternary(
        a=[0 for _ in num_labels],
        b=[100 - v for v in num_labels],
        c=[v for v in num_labels],
        mode="markers+text",
        marker=dict(size=2, color="rgba(0,0,0,0)"),
        text=texts,
        textposition="bottom center",
        textfont=font,
        hoverinfo="skip",
        showlegend=False,
        cliponaxis=False,
    )

    return [fb, fc]


def _build_boundary_traces(boundaries: Dict[str, pd.DataFrame]) -> List[go.Scatterternary]:
    traces: List[go.Scatterternary] = []
    for name, frame in boundaries.items():
        fe = frame["a"].to_numpy() * 100  # Fe%
        cr = frame["b"].to_numpy() * 100  # Cr%
        b = frame["c"].to_numpy() * 100  # B%
        traces.append(
            go.Scatterternary(
                a=b,
                b=fe,
                c=cr,
                mode="lines",
                name=name,
                line=dict(width=2.2),
                hovertemplate=f"{name}<br>B=%{{a:.2f}}%<br>Fe=%{{b:.2f}}%<br>Cr=%{{c:.2f}}%",
                showlegend=True,
            )
        )
    return traces


def build_plot(
    fe_pct,
    b_pct,
    cr_pct,
    enthalpies,
    step_used: float,
    boundary_traces: Optional[List[go.Scatterternary]] = None,
):
    fig = build_ternary_figure(
        combo=("B", "Fe", "Cr"),
        a_vals=b_pct,
        b_vals=fe_pct,
        c_vals=cr_pct,
        enthalpies=enthalpies,
    )
    fig.update_traces(
        selector=dict(type="scatterternary"),
        marker_colorbar=dict(x=1.02, y=0.5, len=0.78, thickness=14, xpad=10),
    )
    fig.update_layout(
        title=dict(text=f"Fe-B-Cr ΔH_mix (step≈{step_used:.4f})"),
        ternary=dict(
            aaxis=dict(title="B", showticklabels=False, ticks="", showgrid=False),
            baxis=dict(title="Fe", showticklabels=False, ticks="", showgrid=False),
            caxis=dict(title="Cr", showticklabels=False, ticks="", showgrid=False),
        ),
        legend=dict(x=1.18, xanchor="left", y=0.5, orientation="v"),
        margin=dict(l=90, r=320, t=90, b=90),
    )
    fig.add_traces(_edge_labels(num_labels=[0, 25, 50, 75, 100]))
    if boundary_traces:
        for trace in boundary_traces:
            fig.add_trace(trace)
    return fig


# -------------------------- CLI -------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Fe-B-Cr enthalpy preview with phase boundaries snapped to the grid.")
    parser.add_argument("--excel", type=Path, default=DEFAULT_DATABASE_PATH, help="Omega workbook path")
    parser.add_argument(
        "--step",
        type=float,
        default=0.001,
        help="Sampling step (fraction, 0-1], default 0.001 (0.1%%); set explicitly if needed.",
    )
    parser.add_argument("--output", type=Path, help="PNG output path (optional)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI when saving")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument(
        "--boundary-csv",
        type=Path,
        default=Path("Data/FeBCr phase diagram line.csv"),
        help="Phase boundary CSV to snap and overlay",
    )
    parser.add_argument("--no-boundaries", action="store_true", help="Disable boundary overlay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_omega_tables(args.excel)

    fe_pct, b_pct, cr_pct, enthalpies, step_used = compute_fe_b_cr(tables, args.step)

    boundary_traces: Optional[List[go.Scatterternary]] = None
    if not args.no_boundaries:
        boundaries = load_boundaries_snapped(args.boundary_csv, grid_step=step_used)
        if boundaries:
            boundary_traces = _build_boundary_traces(boundaries)

    fig = build_plot(fe_pct, b_pct, cr_pct, enthalpies, step_used, boundary_traces)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_plotly_image(fig, args.output)
        print(f"Saved enthalpy preview to {args.output}")
    if args.show or not args.output:
        fig.show()


if __name__ == "__main__":
    main()

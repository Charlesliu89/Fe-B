#!/usr/bin/env python3
"""Render Fe-B-Cr phase boundaries inside a blank equilateral ternary diagram."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enthalpy_config import DEFAULT_FONT_FAMILY, FONT_COLOR, FONT_SIZE
from enthalpy_core import barycentric_to_cartesian


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


def load_phase_boundaries(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """Load each dataset's a/b/c fractions from the CSV."""
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

        boundaries[name] = _ensure_linear_boundary(numeric[["a", "b", "c"]])

    return boundaries


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
    centered = coords - coords.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        axis = np.array([1.0, 0.0, 0.0])
    else:
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
        except np.linalg.LinAlgError:
            axis = np.array([1.0, 0.0, 0.0])
    projection = centered @ axis
    order = np.argsort(projection, kind="mergesort")
    if len(coords) >= 2:
        first_rank = int(np.where(order == 0)[0][0])
        last_rank = int(np.where(order == len(coords) - 1)[0][0])
        if first_rank > last_rank:
            order = order[::-1]
    return coords[order]


def _ensure_linear_boundary(frame: pd.DataFrame) -> pd.DataFrame:
    coords = frame.to_numpy(dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if len(coords) == 0:
        return frame
    coords = _drop_duplicate_rows(coords)
    coords = np.clip(coords, 0.0, 1.0)
    sums = coords.sum(axis=1, keepdims=True)
    valid = sums.squeeze() > 0
    coords[valid] = coords[valid] / sums[valid]
    coords = _monotonic_order(coords)
    return pd.DataFrame(coords, columns=["a", "b", "c"])


def _triangle_vertices():
    vertices = [
        barycentric_to_cartesian((1.0, 0.0, 0.0)),
        barycentric_to_cartesian((0.0, 1.0, 0.0)),
        barycentric_to_cartesian((0.0, 0.0, 1.0)),
    ]
    xs = [v[0] for v in vertices] + [vertices[0][0]]
    ys = [v[1] for v in vertices] + [vertices[0][1]]
    return xs, ys


def _boundary_to_xy(frame: pd.DataFrame):
    coords = frame[["a", "b", "c"]].to_numpy()
    xy = [barycentric_to_cartesian(tuple(row)) for row in coords]
    xs, ys = zip(*xy)
    return xs, ys


def plot_phase_boundaries(boundaries: Dict[str, pd.DataFrame]) -> plt.Figure:
    """Create the equilateral ternary plot with all boundary polylines."""
    plt.rcParams["font.family"] = DEFAULT_FONT_FAMILY
    plt.rcParams["text.color"] = FONT_COLOR

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(right=0.75)

    tri_x, tri_y = _triangle_vertices()
    ax.fill(tri_x, tri_y, facecolor="#fbfbfb", zorder=0)
    ax.plot(tri_x, tri_y, color="black", linewidth=1.5, zorder=1)

    colors = plt.get_cmap("tab20", max(len(boundaries), 1))
    for idx, (name, frame) in enumerate(boundaries.items()):
        xs, ys = _boundary_to_xy(frame)
        ax.plot(xs, ys, label=name, linewidth=2.2, color=colors(idx), zorder=2)

    if boundaries:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.95)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Fe-B-Cr phase boundaries", fontsize=FONT_SIZE + 2, pad=20)

    vertex_labels = [
        ("Fe", barycentric_to_cartesian((1.0, 0.0, 0.0)), (-0.04, -0.035)),
        ("B", barycentric_to_cartesian((0.0, 1.0, 0.0)), (0.02, -0.035)),
        ("Cr", barycentric_to_cartesian((0.0, 0.0, 1.0)), (0.0, 0.03)),
    ]
    for text, (x, y), (dx, dy) in vertex_labels:
        ax.text(x + dx, y + dy, text, fontsize=FONT_SIZE, fontweight="bold")

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fe-B-Cr phase boundaries from a CSV file.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Data/FeBCr phase diagram line.csv"),
        help="Path to the Fe-B-Cr boundary CSV (default: Data/FeBCr phase diagram line.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FeBCr_phase_boundaries.png"),
        help="Image file to save when --save is provided.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI when saving (default: 300).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the plot to --output instead of opening a preview window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    boundaries = load_phase_boundaries(csv_path)
    if not boundaries:
        raise RuntimeError("No usable datasets were found in the CSV.")

    fig = plot_phase_boundaries(boundaries)
    if args.save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        print(f"Saved Fe-B-Cr phase boundary plot to {args.output}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

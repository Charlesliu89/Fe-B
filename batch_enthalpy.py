#!/usr/bin/env python3
"""
Batch enthalpy of mixing visualizer.

Generates ΔH_mix curves/contours by reusing the single-pair calculator:
1) Batch binary line plots (0–100% at 0.1% increments)
2) Batch ternary contour plots (barycentric triangle)
3) Quaternary interactive tetrahedron preview (optional slice export)
4) Custom combination preview with Plotly (hover + draggable labels, optional export)
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import site
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def _extend_sys_path_with_site_packages() -> None:
    """Ensure user/virtual-env site-packages folders are importable."""
    candidates: List[str] = []
    for attr in ("getusersitepackages", "getsitepackages"):
        getter = getattr(site, attr, None)
        if getter is None:
            continue
        try:
            paths = getter()
        except Exception:  # pragma: no cover - defensive
            continue
        if isinstance(paths, str):
            candidates.append(paths)
        else:
            candidates.extend(paths)
    for path in candidates:
        if path and path not in sys.path:
            sys.path.append(path)


def _raise_missing_package(package: str, exc: ImportError) -> None:
    raise ModuleNotFoundError(
        f"Missing required dependency '{package}'. "
        "Please run `python3 -m pip install -r requirements.txt`."
    ) from exc


_extend_sys_path_with_site_packages()

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    _raise_missing_package("pandas", exc)

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - import guard
    _raise_missing_package("plotly", exc)

DEFAULT_FONT_FAMILY = "DejaVu Sans"

# --------------------------------------------------------------------------- #
# Font configuration (edit here to change all text styles)
# --------------------------------------------------------------------------- #

FONT_SIZE = 20
FONT_COLOR = "black"
FONT_WEIGHT = "bold"

PLOTLY_EXPORT = {
    "width": 1280,
    "height": 960,
    "scale": 2,  # boost PNG resolution
}

PLOTLY_BASE_FONT = {
    "family": DEFAULT_FONT_FAMILY,
    "color": FONT_COLOR,
}

PLOTLY_ELEMENT_FONT = {
    **PLOTLY_BASE_FONT,
    "size": FONT_SIZE,
}

# Unified color bar label configuration (Matplotlib + Plotly共享此参数)
COLORBAR_LABEL_CONFIG = {
    # Matplotlib can render the TeX-friendly string; Plotly PNG export works best with HTML.
    "text": r"$\Delta H_{\mathrm{mix}}$ (kJ/mol)",
    "plotly_text": "ΔH_mix (kJ/mol)",
    "rotation_deg": 0,  # clockwise rotation
    "mat_axes_position": (0.5, 1.02),  # x/y inside Matplotlib color bar axes
    "plotly_position": (1.05, 1.0),  # x/y in Plotly paper coordinates
    "plotly_xanchor": "center",
    "plotly_yanchor": "bottom",
    "font_size": 20,
    "font_weight": "bold",
    "font_color": FONT_COLOR,
    "font_family": DEFAULT_FONT_FAMILY,
}


def add_plotly_colorbar_label(fig: "go.Figure") -> None:
    """Annotate a Plotly figure with the shared color bar label style."""
    cfg = COLORBAR_LABEL_CONFIG
    text = cfg.get("plotly_text") or cfg["text"]
    if str(cfg["font_weight"]).lower() == "bold":
        text = f"<b>{text}</b>"
    fig.add_annotation(
        x=cfg["plotly_position"][0],
        y=cfg["plotly_position"][1],
        xref="paper",
        yref="paper",
        text=text,
        textangle=-cfg["rotation_deg"],  # Plotly also uses CCW-positive angles
        showarrow=False,
        xanchor=cfg["plotly_xanchor"],
        yanchor=cfg["plotly_yanchor"],
        font={
            "family": cfg["font_family"],
            "size": cfg["font_size"],
            "color": cfg["font_color"],
        },
    )


def apply_plotly_base_style(fig: "go.Figure") -> None:
    """Ensure Plotly figures use a Unicode-safe font everywhere."""
    fig.update_layout(font=PLOTLY_BASE_FONT)


def write_plotly_image(fig: "go.Figure", target: Path) -> None:
    """Export Plotly figures with consistent resolution/style settings."""
    fig.write_image(str(target), **PLOTLY_EXPORT)


def build_binary_curve(
    calculator, tables, combo: Sequence[str], total_units: int
) -> Tuple[List[float], List[float]]:
    """Compute fractions and enthalpies for a binary combination."""
    fractions = [i / total_units for i in range(total_units + 1)]
    enthalpies: List[float] = []
    for frac_a in fractions:
        frac_b = 1.0 - frac_a
        composition = [(combo[0], frac_a), (combo[1], frac_b)]
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        enthalpies.append(total_enthalpy)
    return fractions, enthalpies


def build_ternary_points(
    calculator,
    tables,
    combo: Sequence[str],
    vectors: Sequence[Tuple[int, ...]],
    total_units: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute ternary mole fraction grids and enthalpies."""
    a_vals: List[float] = []
    b_vals: List[float] = []
    c_vals: List[float] = []
    enthalpies: List[float] = []
    for vector in vectors:
        fractions = fractions_from_vector(vector, total_units)
        composition = list(zip(combo, fractions))
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        a_vals.append(fractions[0] * 100)
        b_vals.append(fractions[1] * 100)
        c_vals.append(fractions[2] * 100)
        enthalpies.append(total_enthalpy)
    return a_vals, b_vals, c_vals, enthalpies


def build_binary_figure(
    combo: Sequence[str], fractions: Sequence[float], enthalpies: Sequence[float]
) -> "go.Figure":
    """Return a styled Plotly figure for a binary mixture."""
    fig = go.Figure(
        go.Scatter(
            x=[f * 100 for f in fractions],
            y=enthalpies,
            mode="lines+markers",
            hovertemplate=(
                f"{combo[0]}=%{{x:.3f}}%\n"
                f"{combo[1]}=%{{customdata:.3f}}%\n"
                "ΔH=%{y:.5f} kJ/mol"
            ),
            customdata=[(1.0 - f) * 100 for f in fractions],
        )
    )
    fig.update_layout(
        title=dict(
            text=r"Binary $\Delta H_{\mathrm{mix}}$: " + "-".join(combo),
            font=PLOTLY_ELEMENT_FONT,
        ),
        xaxis=dict(title=dict(text=f"{combo[0]} atomic %", font=PLOTLY_ELEMENT_FONT)),
        yaxis=dict(title=dict(text=r"$\Delta H_{\mathrm{mix}}$ (kJ/mol)", font=PLOTLY_ELEMENT_FONT)),
        template="plotly_white",
    )
    apply_plotly_base_style(fig)
    return fig


def build_ternary_figure(
    combo: Sequence[str],
    a_vals: Sequence[float],
    b_vals: Sequence[float],
    c_vals: Sequence[float],
    enthalpies: Sequence[float],
) -> "go.Figure":
    """Return a styled Plotly ternary scatter figure."""
    fig = go.Figure(
        go.Scatterternary(
            a=a_vals,
            b=b_vals,
            c=c_vals,
            mode="markers",
            marker=dict(
                size=6,
                color=enthalpies,
                colorscale="Viridis",
                colorbar=dict(
                    thickness=15,
                    len=0.75,
                    title=dict(
                        text=COLORBAR_LABEL_CONFIG.get("plotly_text"),
                        font=PLOTLY_ELEMENT_FONT,
                        side="top",
                    ),
                    xpad=40,
                ),
            ),
            hovertemplate=(
                f"{combo[0]}=%{{a:.2f}}%<br>"
                f"{combo[1]}=%{{b:.2f}}%<br>"
                f"{combo[2]}=%{{c:.2f}}%<br>"
                "ΔH=%{marker.color:.5f} kJ/mol"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Ternary ΔH<sub>mix</sub>: {'-'.join(combo)}",
            font=PLOTLY_ELEMENT_FONT,
        ),
        ternary=dict(
            sum=100,
            aaxis=dict(title=dict(text=combo[0], font=PLOTLY_ELEMENT_FONT)),
            baxis=dict(title=dict(text=combo[1], font=PLOTLY_ELEMENT_FONT)),
            caxis=dict(title=dict(text=combo[2], font=PLOTLY_ELEMENT_FONT)),
        ),
        margin=dict(l=60, r=200, t=80, b=80),
        template="plotly_white",
    )
    apply_plotly_base_style(fig)
    return fig


def barycentric_to_cartesian(fractions: Sequence[float]) -> Tuple[float, float, float]:
    """Map 4-component barycentric coordinates to a regular tetrahedron in 3D space."""

    if len(fractions) != 4:
        raise ValueError("Quaternary barycentric coordinates require four fractions.")

    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, math.sqrt(3) / 2.0, 0.0),
        (0.5, math.sqrt(3) / 6.0, math.sqrt(2.0 / 3.0)),
    ]
    x = sum(f * v[0] for f, v in zip(fractions, vertices))
    y = sum(f * v[1] for f, v in zip(fractions, vertices))
    z = sum(f * v[2] for f, v in zip(fractions, vertices))
    return x, y, z


def build_quaternary_points(
    calculator,
    tables,
    combo: Sequence[str],
    vectors: Sequence[Tuple[int, ...]],
    total_units: int,
) -> Tuple[List[float], List[float], List[float], List[float], List[Tuple[float, ...]]]:
    """Compute 3D coordinates, enthalpies, and raw fraction tuples for quaternary alloys."""

    x_vals: List[float] = []
    y_vals: List[float] = []
    z_vals: List[float] = []
    enthalpies: List[float] = []
    fractions_list: List[Tuple[float, ...]] = []

    for vector in vectors:
        fractions = fractions_from_vector(vector, total_units)
        composition = list(zip(combo, fractions))
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        x, y, z = barycentric_to_cartesian(fractions)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
        enthalpies.append(total_enthalpy)
        fractions_list.append(fractions)

    return x_vals, y_vals, z_vals, enthalpies, fractions_list


def build_quaternary_figure(
    combo: Sequence[str],
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    z_vals: Sequence[float],
    enthalpies: Sequence[float],
    fractions: Sequence[Tuple[float, ...]],
) -> "go.Figure":
    """Return a rotatable Plotly tetrahedron with color-coded ΔH_mix distribution."""

    vertex_labels = []
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, math.sqrt(3) / 2.0, 0.0),
        (0.5, math.sqrt(3) / 6.0, math.sqrt(2.0 / 3.0)),
    ]
    for idx, name in enumerate(combo):
        vertex_labels.append(
            go.Scatter3d(
                x=[vertices[idx][0]],
                y=[vertices[idx][1]],
                z=[vertices[idx][2]],
                mode="text",
                text=[f"<b>{name}</b>"],
                textfont=dict(**PLOTLY_ELEMENT_FONT, size=FONT_SIZE + 2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    edge_traces = []
    for a_idx, b_idx in edges:
        edge_traces.append(
            go.Scatter3d(
                x=[vertices[a_idx][0], vertices[b_idx][0]],
                y=[vertices[a_idx][1], vertices[b_idx][1]],
                z=[vertices[a_idx][2], vertices[b_idx][2]],
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    hover_lines = []
    for frac in fractions:
        hover_lines.append(
            "<br>".join(
                f"{elem}={value * 100:.2f}%" for elem, value in zip(combo, frac)
            )
        )

    shell = go.Mesh3d(
        x=[v[0] for v in vertices],
        y=[v[1] for v in vertices],
        z=[v[2] for v in vertices],
        i=[0, 0, 0, 1],
        j=[1, 2, 3, 2],
        k=[2, 3, 1, 3],
        opacity=0.15,
        color="lightgray",
        flatshading=True,
        name="composition space",
        hoverinfo="skip",
        showscale=False,
    )

    scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode="markers",
        marker=dict(
            size=4,
            color=enthalpies,
            colorscale="Viridis",
            opacity=0.75,
            colorbar=dict(
                title=dict(text=COLORBAR_LABEL_CONFIG.get("plotly_text"), font=PLOTLY_ELEMENT_FONT),
                len=0.6,
            ),
        ),
        hovertemplate="%{text}<br>ΔH=%{marker.color:.5f} kJ/mol",
        text=hover_lines,
    )

    fig = go.Figure([shell, scatter, *edge_traces, *vertex_labels])
    fig = go.Figure([scatter, *edge_traces, *vertex_labels])
    fig.update_layout(
        title=dict(
            text=f"Quaternary ΔH<sub>mix</sub>: {'-'.join(combo)}", font=PLOTLY_ELEMENT_FONT
        ),
        scene=dict(
            xaxis=dict(
                title="X",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                backgroundcolor="white",
            ),
            yaxis=dict(
                title="Y",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                backgroundcolor="white",
            ),
            zaxis=dict(
                title="Z",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                backgroundcolor="white",
            ),
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        template="plotly_white",
    )
    apply_plotly_base_style(fig)
    add_plotly_colorbar_label(fig)
    return fig

# --------------------------------------------------------------------------- #
# Sampling configuration
# --------------------------------------------------------------------------- #

BINARY_STEP = 0.001  # 0.1%
TERNARY_STEP = 0.01  # 1%
QUATERNARY_STEP = 0.05  # 5% for manageable preview density
QUATERNARY_MIN_STEP = 0.01  # allow finer previews when desired
BATCH_CHUNK_SIZE = 100

# --------------------------------------------------------------------------- #
# Calculator module loading
# --------------------------------------------------------------------------- #


def load_calculator_module(script_path: Path):
    """Dynamically load the interactive calculator to reuse its helper functions."""
    if not script_path.exists():
        raise FileNotFoundError(f"Calculator script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("enthalpy_calculator", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load calculator module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


# --------------------------------------------------------------------------- #
# Composition utilities
# --------------------------------------------------------------------------- #


def normalize_step(step: float) -> Tuple[int, float]:
    """Return total units and the adjusted step so that 1 = step * total_units."""
    if step <= 0 or step > 1:
        raise ValueError("Step must be within (0, 1].")
    total_units = round(1.0 / step)
    actual_step = 1.0 / total_units
    if abs(actual_step - step) > 1e-9:
        print(
            f"[warning] Step {step} adjusted to {actual_step:.10f} to evenly divide 1.0.",
            file=sys.stderr,
        )
    return total_units, actual_step


def build_fraction_vectors(count: int, total_units: int) -> List[Tuple[int, ...]]:
    """Enumerate integer solutions that sum to total_units (allowing zeros)."""

    def recurse(index: int, remaining: int, current: List[int]) -> Iterator[Tuple[int, ...]]:
        if index == count - 1:
            current.append(remaining)
            yield tuple(current)
            current.pop()
            return
        for value in range(0, remaining + 1):
            current.append(value)
            yield from recurse(index + 1, remaining - value, current)
            current.pop()

    return list(recurse(0, total_units, []))


def fractions_from_vector(vector: Sequence[int], total_units: int) -> Tuple[float, ...]:
    """Convert integer counts to normalized mole fractions."""
    if total_units == 0:
        raise ValueError("Total units must be positive.")
    return tuple(value / total_units for value in vector)


# --------------------------------------------------------------------------- #
def combo_supported(calculator, tables, combo: Sequence[str]) -> bool:
    """Return True if every pair within combo has Ω data."""
    try:
        for elem_a, elem_b in combinations(combo, 2):
            calculator.lookup_omegas(tables, elem_a, elem_b)
        return True
    except KeyError:
        return False


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def preview_and_maybe_save(fig: go.Figure, default_path: Path) -> None:
    config = {"editable": True, "edits": {"annotationPosition": True}}
    fig.show(config=config)
    save = input(f"Save figure to {default_path}? (y/n): ").strip().lower()
    if save == "y":
        default_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_plotly_image(fig, default_path)
            print(f"Saved PNG to {default_path}")
        except Exception as exc:  # pylint: disable=broad-except
            alt_path = default_path.with_suffix(".html")
            fig.write_html(str(alt_path))
            print(f"PNG export failed ({exc}); saved interactive HTML to {alt_path}")


def _slice_quaternary_data(
    combo: Sequence[str],
    fractions: Sequence[Tuple[float, ...]],
    enthalpies: Sequence[float],
    fixed_element: str,
    fixed_fraction: float,
    tolerance: float,
) -> Tuple[Sequence[str], List[float], List[float], List[float], List[float]]:
    """Extract a ternary slice by fixing one element's fraction (within tolerance)."""

    if fixed_fraction <= 0 or fixed_fraction >= 1:
        raise ValueError("Fixed fraction must be within (0, 1).")

    if fixed_element not in combo:
        raise ValueError(f"{fixed_element} is not part of the chosen quaternary system.")

    fixed_index = combo.index(fixed_element)
    remaining_elements = [elem for elem in combo if elem != fixed_element]

    a_vals: List[float] = []
    b_vals: List[float] = []
    c_vals: List[float] = []
    enthalpy_slice: List[float] = []

    for frac, enthalpy in zip(fractions, enthalpies):
        if abs(frac[fixed_index] - fixed_fraction) > tolerance:
            continue
        remainder = 1.0 - fixed_fraction
        if remainder <= 0:
            continue

        renormalized = [value / remainder for idx, value in enumerate(frac) if idx != fixed_index]
        if len(renormalized) != 3:
            continue

        a_vals.append(renormalized[0] * 100)
        b_vals.append(renormalized[1] * 100)
        c_vals.append(renormalized[2] * 100)
        enthalpy_slice.append(enthalpy)

    return remaining_elements, a_vals, b_vals, c_vals, enthalpy_slice


# --------------------------------------------------------------------------- #
# Parallel worker helpers
# --------------------------------------------------------------------------- #

_WORKER_CALCULATOR = None
_WORKER_TABLES: Dict[str, object] | None = None


def _parallel_initializer(calculator_path: str, tables: Dict[str, object]) -> None:
    """Load calculator/tables once per worker process."""
    global _WORKER_CALCULATOR, _WORKER_TABLES
    _WORKER_CALCULATOR = load_calculator_module(Path(calculator_path))
    _WORKER_TABLES = tables


def _get_worker_state():
    if _WORKER_CALCULATOR is None or _WORKER_TABLES is None:
        raise RuntimeError("Worker state not initialized.")
    return _WORKER_CALCULATOR, _WORKER_TABLES


def _save_binary_figure(combo: Sequence[str], fractions, enthalpies, output_dir: Path) -> None:
    fig = build_binary_figure(combo, fractions, enthalpies)
    target = ensure_directory(output_dir) / f"{combo[0]}-{combo[1]}.png"
    write_plotly_image(fig, target)


def _save_ternary_figure(
    combo: Sequence[str], a_vals, b_vals, c_vals, enthalpies, output_dir: Path
) -> None:
    fig = build_ternary_figure(combo, a_vals, b_vals, c_vals, enthalpies)
    target = ensure_directory(output_dir) / f"{combo[0]}-{combo[1]}-{combo[2]}.png"
    write_plotly_image(fig, target)


def _binary_worker_task(combo: Sequence[str], total_units: int):
    calculator, tables = _get_worker_state()
    fractions, enthalpies = build_binary_curve(calculator, tables, combo, total_units)
    return combo, fractions, enthalpies


def _ternary_worker_task(
    combo: Sequence[str],
    vectors: Sequence[Tuple[int, ...]],
    total_units: int,
):
    calculator, tables = _get_worker_state()
    a_vals, b_vals, c_vals, enthalpies = build_ternary_points(
        calculator, tables, combo, vectors, total_units
    )
    return combo, a_vals, b_vals, c_vals, enthalpies


# --------------------------------------------------------------------------- #
# Core batch processing
# --------------------------------------------------------------------------- #


def run_batch(
    calculator,
    tables: Dict[str, pd.DataFrame],
    component_count: int,
    elements: Sequence[str],
    output_path: Path,
    workers: int = 1,
    chunk_size: int = BATCH_CHUNK_SIZE,
) -> None:
    """Generate one plot per element combination for the chosen component count."""
    if component_count not in {2, 3, 4}:
        raise ValueError("Only 2-, 3-, or 4-component alloys are supported.")

    available_elements = [el for el in tables[calculator.OMEGA_SHEETS[0]].index if el in elements]
    if len(available_elements) < component_count:
        raise ValueError("Not enough elements to form the requested combinations.")

    # Binary alloys require 0.1% increments regardless of the CLI step.
    total_units, actual_step = normalize_step(BINARY_STEP if component_count == 2 else TERNARY_STEP)
    vectors: Optional[List[Tuple[int, ...]]] = None
    if component_count == 3:
        vectors = build_fraction_vectors(component_count, total_units)
        if not vectors:
            raise ValueError("No feasible compositions were generated with the provided step.")

    ensure_directory(output_path)
    processed = 0
    skipped = 0
    worker_count = 0
    supported_combos: List[Tuple[str, ...]] = []

    for combo in combinations(available_elements, component_count):
        if not combo_supported(calculator, tables, combo):
            skipped += 1
            continue

        if component_count in {2, 3}:
            supported_combos.append(combo)
        else:
            print(
                f"[info] Skipping {combo}: use menu option 3 for quaternary preview/slices."
            )
            skipped += 1
            continue

    if not supported_combos:
        print(
            f"[summary] step={actual_step:.4f}, workers={worker_count}: "
            f"{processed} combinations plotted, {skipped} skipped.",
            file=sys.stderr,
        )
        return

    worker_count = min(max(1, workers), len(supported_combos))

    def process_chunk(chunk: List[Tuple[str, ...]], use_parallel: bool) -> int:
        completed = 0
        if use_parallel and len(chunk) > 1:
            calculator_path = getattr(calculator, "__file__", None)
            if not calculator_path:
                raise RuntimeError("Calculator module path is required for parallel execution.")
            init_args = (str(calculator_path), tables)
            with ProcessPoolExecutor(
                max_workers=min(worker_count, len(chunk)),
                initializer=_parallel_initializer,
                initargs=init_args,
            ) as executor:
                futures = []
                for combo in chunk:
                    if component_count == 2:
                        futures.append(executor.submit(_binary_worker_task, combo, total_units))
                    else:
                        assert vectors is not None
                        futures.append(
                            executor.submit(_ternary_worker_task, combo, vectors, total_units)
                        )
                for future in as_completed(futures):
                    result = future.result()
                    if component_count == 2:
                        combo, fractions, enthalpies = result
                        _save_binary_figure(combo, fractions, enthalpies, output_path / "binary")
                    else:
                        combo, a_vals, b_vals, c_vals, enthalpies = result
                        _save_ternary_figure(
                            combo, a_vals, b_vals, c_vals, enthalpies, output_path / "ternary"
                        )
                    completed += 1
            return completed

        for combo in chunk:
            if component_count == 2:
                fractions, enthalpies = build_binary_curve(calculator, tables, combo, total_units)
                _save_binary_figure(combo, fractions, enthalpies, output_path / "binary")
            else:
                assert vectors is not None
                a_vals, b_vals, c_vals, enthalpies = build_ternary_points(
                    calculator, tables, combo, vectors, total_units
                )
                _save_ternary_figure(
                    combo, a_vals, b_vals, c_vals, enthalpies, output_path / "ternary"
                )
            completed += 1
        return completed

    export_all_remaining = False
    index = 0
    total = len(supported_combos)
    while index < total:
        current_chunk_size = total - index if export_all_remaining else chunk_size
        chunk = supported_combos[index : index + current_chunk_size]
        processed += process_chunk(chunk, use_parallel=worker_count > 1)
        index += len(chunk)

        if index >= total:
            break

        while True:
            decision = input(
                f"Exported {index}/{total} images. "
                f"Press 'c' to continue next {chunk_size}, 'a' to export all remaining, "
                "or 'q' to stop: "
            ).strip().lower()
            if decision in {"c", "", "continue"}:
                break
            if decision in {"a", "all"}:
                export_all_remaining = True
                break
            if decision in {"q", "quit", "stop"}:
                index = total  # exit outer loop
                break
            print("Please enter 'c', 'a', or 'q'.")

    print(
        f"[summary] step={actual_step:.4f}, workers={worker_count}: "
        f"{processed} combinations plotted, {skipped} skipped.",
        file=sys.stderr,
    )


def handle_custom_plot(calculator, tables, output_dir: Path) -> None:
    prompt = (
        "Enter element symbols separated by commas (e.g., Fe,B or Fe,B,Ni)"
        " or 'b' to return to the main menu: "
    )
    custom_dir = ensure_directory(output_dir / "custom")

    while True:
        raw = input(prompt).strip()
        if not raw:
            print("No elements entered. Provide symbols or 'b' to return.")
            continue
        if raw.lower() in {"b", "back", "r", "return"}:
            print("Returning to the main menu.")
            return

        symbols = [calculator.normalize_symbol(part) for part in re.split(r"[\\s,]+", raw) if part]
        unique_elements = []
        for symbol in symbols:
            if symbol not in unique_elements:
                unique_elements.append(symbol)
        if len(unique_elements) < 2 or len(unique_elements) > 4:
            print("Please provide between 2 and 4 unique elements.")
            continue
        for element in unique_elements:
            if element not in tables[calculator.OMEGA_SHEETS[0]].index:
                print(f"Element {element} is not available in the database.")
                break
        else:
            try:
                fig, filename = build_custom_plot(calculator, tables, unique_elements)
            except ValueError as exc:
                print(exc)
                continue
            preview_and_maybe_save(fig, custom_dir / filename)


def handle_quaternary_preview(calculator, tables, output_dir: Path) -> None:
    """Interactive menu entry for quaternary ΔH_mix preview and slice export."""

    prompt = "Enter four element symbols separated by commas (or 'b' to return): "
    quaternary_dir = ensure_directory(output_dir / "quaternary")

    while True:
        raw = input(prompt).strip()
        if not raw:
            print("No elements entered. Please provide four symbols or 'b' to return.")
            continue
        if raw.lower() in {"b", "back", "r", "return"}:
            print("Returning to the main menu.")
            return

        elements = [calculator.normalize_symbol(part) for part in re.split(r"[\\s,]+", raw) if part]
        unique_elements: List[str] = []
        for element in elements:
            if element not in unique_elements:
                unique_elements.append(element)

        if len(unique_elements) != 4:
            print("Please provide exactly four unique elements.")
            continue

        if not combo_supported(calculator, tables, unique_elements):
            print("Ω data is incomplete for at least one element pair; choose a different set.")
            continue

        density_raw = input(
            "Preview step size in % (press Enter for default 5%): "
        ).strip()
        preview_step = QUATERNARY_STEP
        if density_raw:
            try:
                preview_step = float(density_raw) / 100.0
                if preview_step <= 0:
                    raise ValueError
                if preview_step < QUATERNARY_MIN_STEP:
                    print(
                        f"Clamping to minimum step {QUATERNARY_MIN_STEP * 100:.1f}% for stability."
                    )
                    preview_step = QUATERNARY_MIN_STEP
            except ValueError:
                print("Invalid percentage. Using default 5% step.")
                preview_step = QUATERNARY_STEP

        total_units, actual_step = normalize_step(preview_step)
        total_units, actual_step = normalize_step(QUATERNARY_STEP)
        vectors = build_fraction_vectors(4, total_units)
        if not vectors:
            print(
                f"No feasible compositions generated for step={actual_step:.3f}. "
                "Adjust QUATERNARY_STEP if needed."
            )
            continue

        print(
            f"Sampling {len(vectors)} compositions with {actual_step * 100:.2f}% increments."
        )

        combo = tuple(unique_elements)
        x_vals, y_vals, z_vals, enthalpies, fractions = build_quaternary_points(
            calculator, tables, combo, vectors, total_units
        )
        fig = build_quaternary_figure(combo, x_vals, y_vals, z_vals, enthalpies, fractions)
        fig.show(config={"displaylogo": False, "displayModeBar": True})

        while True:
            slice_choice = input(
                "Export ternary projection by fixing one element (e.g., Fe=25)? (y/n): "
            ).strip().lower()
            if slice_choice in {"n", "no"}:
                break
            if slice_choice not in {"y", "yes"}:
                print("Please answer 'y' or 'n'.")
                continue

            slice_raw = input("Enter element=fraction% to fix (e.g., Fe=25): ").strip()
            match = re.match(r"([A-Za-z]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)", slice_raw)
            if not match:
                print("Invalid format. Use Element=number (percentage).")
                continue

            element = calculator.normalize_symbol(match.group(1))
            fraction_percent = float(match.group(2))
            if fraction_percent <= 0 or fraction_percent >= 100:
                print("Fraction must be between 0 and 100 (exclusive).")
                continue

            try:
                remaining_elements, a_vals, b_vals, c_vals, enthalpy_slice = _slice_quaternary_data(
                    combo,
                    fractions,
                    enthalpies,
                    element,
                    fraction_percent / 100.0,
                    tolerance=actual_step / 2,
                    tolerance=QUATERNARY_STEP / 2,
                )
            except ValueError as exc:
                print(exc)
                continue

            if not enthalpy_slice:
                print(
                    "No compositions matched that fixed fraction. "
                    "Consider relaxing QUATERNARY_STEP."
                )
                continue

            slice_fig = build_ternary_figure(
                remaining_elements, a_vals, b_vals, c_vals, enthalpy_slice
            )
            filename = f"{'-'.join(combo)}_{element}{fraction_percent:.0f}.png"
            preview_and_maybe_save(slice_fig, quaternary_dir / filename)
            break


def build_custom_plot(
    calculator,
    tables,
    elements: Sequence[str],
) -> Tuple[go.Figure, str]:
    """Create a Plotly figure for 2–3 elements without interactive input."""
    if len(elements) < 2 or len(elements) > 4:
        raise ValueError("Please provide between 2 and 4 unique elements.")
    for element in elements:
        if element not in tables[calculator.OMEGA_SHEETS[0]].index:
            raise ValueError(f"Element {element} is not available in the database.")

    component_count = len(elements)
    step_value = BINARY_STEP if component_count == 2 else TERNARY_STEP
    total_units, _ = normalize_step(step_value if component_count < 4 else QUATERNARY_STEP)
    combo = tuple(elements)

    if component_count == 2:
        fractions, enthalpies = build_binary_curve(calculator, tables, combo, total_units)
        fig = build_binary_figure(combo, fractions, enthalpies)
        return fig, f"{combo[0]}-{combo[1]}.png"

    if component_count == 3:
        vectors = build_fraction_vectors(3, total_units)
        a_vals, b_vals, c_vals, enthalpies = build_ternary_points(
            calculator, tables, combo, vectors, total_units
        )
        fig = build_ternary_figure(combo, a_vals, b_vals, c_vals, enthalpies)
        return fig, f"{combo[0]}-{combo[1]}-{combo[2]}.png"

    if component_count == 4:
        vectors = build_fraction_vectors(4, total_units)
        if not vectors:
            raise ValueError("No feasible quaternary compositions were generated.")
        x_vals, y_vals, z_vals, enthalpies, fractions = build_quaternary_points(
            calculator, tables, combo, vectors, total_units
        )
        fig = build_quaternary_figure(combo, x_vals, y_vals, z_vals, enthalpies, fractions)
        return fig, f"{combo[0]}-{combo[1]}-{combo[2]}-{combo[3]}.png"

    raise ValueError("Unsupported component count.")


def generate_custom_plots(
    calculator,
    tables,
    combos: Sequence[Sequence[str]],
    output_dir: Path,
) -> List[Path]:
    """Programmatically generate custom plots for given combos."""
    custom_dir = ensure_directory(output_dir / "custom")
    saved: List[Path] = []

    for elements in combos:
        try:
            fig, filename = build_custom_plot(calculator, tables, elements)
        except ValueError as exc:
            print(f"[auto] Skipping {elements}: {exc}")
            continue

        target = custom_dir / filename
        try:
            write_plotly_image(fig, target)
            print(f"[auto] Saved PNG to {target}")
            saved.append(target)
        except Exception as exc:  # pylint: disable=broad-except
            alt_path = target.with_suffix(".html")
            fig.write_html(str(alt_path))
            print(f"[auto] PNG export failed ({exc}); saved interactive HTML to {alt_path}")
            saved.append(alt_path)

    return saved


# --------------------------------------------------------------------------- #
# Argument parsing & entry point
# --------------------------------------------------------------------------- #


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch enthalpy plot generator.")
    parser.add_argument(
        "--calculator",
        type=Path,
        default=Path(__file__).with_name("enthalpy of mixing.py"),
        help="Path to the interactive calculator script.",
    )
    parser.add_argument(
        "--excel-db",
        type=Path,
        default=None,
        help="Path to the Omega matrices workbook (defaults to calculator's setting).",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        help="Optional whitelist of element symbols (defaults to all available).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/plots"),
        help="Directory to store generated plots (default: Data/plots).",
    )
    parser.add_argument(
        "--auto-combo",
        action="append",
        help=(
            "Non-interactive mode: provide a comma-separated element list "
            "(e.g., 'Fe,B,Ni'); can be repeated to render multiple combos then exit."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes to use (default: CPU count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calculator = load_calculator_module(args.calculator)

    excel_path = (
        Path(args.excel_db)
        if args.excel_db
        else Path(calculator.DEFAULT_DATABASE_PATH)  # type: ignore[attr-defined]
    )
    tables = calculator.load_omega_tables(excel_path)

    element_pool = (
        [calculator.normalize_symbol(sym) for sym in args.elements]
        if args.elements
        else list(tables[calculator.OMEGA_SHEETS[0]].index)
    )

    if args.auto_combo:
        combos: List[List[str]] = []
        for raw in args.auto_combo:
            parts = [calculator.normalize_symbol(part) for part in re.split(r"[\\s,]+", raw) if part]
            if not parts:
                print(f"[auto] Skipping empty combo input: {raw!r}")
                continue
            unique: List[str] = []
            for symbol in parts:
                if symbol not in unique:
                    unique.append(symbol)
            combos.append(unique)

        if combos:
            generate_custom_plots(calculator, tables, combos, args.output_dir)
        else:
            print("[auto] No valid combinations were provided.")
        return

    while True:
        print("\n=== Enthalpy Plot Menu ===")
        print("1) Batch binary ΔH_mix curves")
        print("2) Batch ternary ΔH_mix contour plots")
        print("3) Quaternary ΔH_mix tetrahedron (preview + slice export)")
        print("4) Custom combination plot")
        print("q) Quit")
        choice = input("Select an option: ").strip().lower()

        if choice == "1":
            try:
                run_batch(
                    calculator=calculator,
                    tables=tables,
                    component_count=2,
                    elements=element_pool,
                    output_path=args.output_dir,
                    workers=args.workers,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Binary plotting failed: {exc}")
        elif choice == "2":
            try:
                run_batch(
                    calculator=calculator,
                    tables=tables,
                    component_count=3,
                    elements=element_pool,
                    output_path=args.output_dir,
                    workers=args.workers,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Ternary plotting failed: {exc}")
        elif choice == "3":
            try:
                handle_quaternary_preview(calculator, tables, args.output_dir)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Quaternary preview failed: {exc}")
        elif choice == "4":
            handle_custom_plot(calculator, tables, args.output_dir)
        elif choice in {"q", "quit", "exit"}:
            print("Bye.")
            break
        else:
            print("Invalid selection. Please choose 1–4 or q.")


if __name__ == "__main__":
    main()

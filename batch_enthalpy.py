#!/usr/bin/env python3
"""
Batch enthalpy of mixing visualizer.

Generates ΔH_mix curves/contours by reusing the single-pair calculator:
1) Batch binary line plots (0–100% at 0.1% increments)
2) Batch ternary contour plots (barycentric triangle)
3) (Reserved) Quaternary – currently skipped
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
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
except ImportError as exc:  # pragma: no cover - import guard
    _raise_missing_package("matplotlib", exc)

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - import guard
    _raise_missing_package("pandas", exc)

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - import guard
    _raise_missing_package("plotly", exc)

DEFAULT_FONT_FAMILY = "DejaVu Sans"

# Ensure Δ / Greek letters render correctly across Matplotlib outputs
plt.rcParams.update(
    {
        "font.family": DEFAULT_FONT_FAMILY,
        "axes.unicode_minus": False,
    }
)

# --------------------------------------------------------------------------- #
# Font configuration (edit here to change all text styles)
# --------------------------------------------------------------------------- #

FONT_SIZE = 20
FONT_COLOR = "black"
FONT_WEIGHT = "bold"

PLOTLY_BASE_FONT = {
    "family": DEFAULT_FONT_FAMILY,
    "color": FONT_COLOR,
}

ELEMENT_LABEL_FONT = {
    "fontsize": FONT_SIZE,
    "fontweight": FONT_WEIGHT,
    "fontfamily": DEFAULT_FONT_FAMILY,
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
    "plotly_text": "ΔH<sub>mix</sub> (kJ/mol)",
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


def add_matplotlib_colorbar_label(cbar) -> None:
    """Apply the shared color bar label style to a Matplotlib color bar."""
    cfg = COLORBAR_LABEL_CONFIG
    cbar.ax.text(
        cfg["mat_axes_position"][0],
        cfg["mat_axes_position"][1],
        cfg["text"],
        transform=cbar.ax.transAxes,
        rotation=-cfg["rotation_deg"],  # Matplotlib uses CCW-positive angles
        ha="center",
        va="bottom",
        fontsize=cfg["font_size"],
        fontweight=cfg["font_weight"],
        color=cfg["font_color"],
    )


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

# --------------------------------------------------------------------------- #
# Sampling configuration
# --------------------------------------------------------------------------- #

BINARY_STEP = 0.001  # 0.1%
TERNARY_STEP = 0.01  # 1%

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


def barycentric_to_cartesian(fractions: Sequence[float]) -> Tuple[float, float]:
    """Map ternary fractions (A,B,C) to 2D coordinates inside an equilateral triangle."""
    if len(fractions) != 3:
        raise ValueError("Ternary barycentric conversion requires exactly 3 fractions.")
    a, b, c = fractions
    x = b + 0.5 * c
    y = (math.sqrt(3) / 2.0) * c
    return x, y


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


def save_binary_plot(combo: Sequence[str], fractions: List[float], enthalpies: List[float], out_dir: Path) -> None:
    path = ensure_directory(out_dir) / f"{combo[0]}-{combo[1]}.png"
    plt.figure(figsize=(6, 4))
    plt.plot([f * 100 for f in fractions], enthalpies, lw=1.5)
    plt.xlabel(f"{combo[0]} atomic %")
    plt.ylabel(r"$\Delta H_{\mathrm{mix}}$ (kJ/mol)")
    plt.title(r"Binary $\Delta H_{\mathrm{mix}}$: " + "-".join(combo))
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_ternary_plot(
    combo: Sequence[str],
    xs: List[float],
    ys: List[float],
    values: List[float],
    out_dir: Path,
) -> None:
    path = ensure_directory(out_dir) / f"{combo[0]}-{combo[1]}-{combo[2]}.png"
    plt.figure(figsize=(6, 5.5))
    triang = mtri.Triangulation(xs, ys)
    mesh = plt.tripcolor(triang, values, shading="gouraud", cmap="viridis")
    cbar = plt.colorbar(mesh)
    add_matplotlib_colorbar_label(cbar)
    combo_label = "-".join(combo)
    plt.title(r"Ternary $\Delta H_{\mathrm{mix}}$: " + combo_label)
    # annotate corners
    corners = {
        combo[0]: (0.0, 0.0),
        combo[1]: (1.0, 0.0),
        combo[2]: (0.5, math.sqrt(3) / 2.0),
    }
    for name, (x, y) in corners.items():
        plt.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            **ELEMENT_LABEL_FONT,
        )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def preview_and_maybe_save(fig: go.Figure, default_path: Path) -> None:
    config = {"editable": True, "edits": {"annotationPosition": True}}
    fig.show(config=config)
    save = input(f"Save figure to {default_path}? (y/n): ").strip().lower()
    if save == "y":
        default_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(default_path))
            print(f"Saved PNG to {default_path}")
        except Exception as exc:  # pylint: disable=broad-except
            alt_path = default_path.with_suffix(".html")
            fig.write_html(str(alt_path))
            print(f"PNG export failed ({exc}); saved interactive HTML to {alt_path}")


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


def _binary_worker_task(combo: Sequence[str], total_units: int):
    calculator, tables = _get_worker_state()
    fractions = [i / total_units for i in range(total_units + 1)]
    enthalpies: List[float] = []
    for frac_a in fractions:
        frac_b = 1.0 - frac_a
        composition = [(combo[0], frac_a), (combo[1], frac_b)]
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        enthalpies.append(total_enthalpy)
    return combo, fractions, enthalpies


def _ternary_worker_task(
    combo: Sequence[str],
    vectors: Sequence[Tuple[int, ...]],
    total_units: int,
):
    calculator, tables = _get_worker_state()
    xs: List[float] = []
    ys: List[float] = []
    values: List[float] = []
    for vector in vectors:
        fractions = fractions_from_vector(vector, total_units)
        composition = list(zip(combo, fractions))
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        x, y = barycentric_to_cartesian(fractions)
        xs.append(x)
        ys.append(y)
        values.append(total_enthalpy)
    return combo, xs, ys, values


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

        if component_count == 2:
            supported_combos.append(combo)
        elif component_count == 3:
            supported_combos.append(combo)
        else:
            print(f"[info] Skipping {combo}: quaternary plotting not implemented.")
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

    if worker_count == 1:
        for combo in supported_combos:
            if component_count == 2:
                plot_binary_combination(
                    calculator, tables, combo, total_units, output_path / "binary"
                )
            else:
                assert vectors is not None
                plot_ternary_combination(
                    calculator,
                    tables,
                    combo,
                    vectors,
                    total_units,
                    output_path / "ternary",
                )
            processed += 1
    else:
        calculator_path = getattr(calculator, "__file__", None)
        if not calculator_path:
            raise RuntimeError("Calculator module path is required for parallel execution.")
        init_args = (str(calculator_path), tables)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_parallel_initializer,
            initargs=init_args,
        ) as executor:
            futures = []
            for combo in supported_combos:
                if component_count == 2:
                    futures.append(executor.submit(_binary_worker_task, combo, total_units))
                else:
                    assert vectors is not None
                    futures.append(executor.submit(_ternary_worker_task, combo, vectors, total_units))
            for future in as_completed(futures):
                result = future.result()
                if component_count == 2:
                    combo, fractions, enthalpies = result
                    save_binary_plot(combo, fractions, enthalpies, output_path / "binary")
                else:
                    combo, xs, ys, values = result
                    save_ternary_plot(combo, xs, ys, values, output_path / "ternary")
                processed += 1

    print(
        f"[summary] step={actual_step:.4f}, workers={worker_count}: "
        f"{processed} combinations plotted, {skipped} skipped.",
        file=sys.stderr,
    )


def plot_binary_combination(calculator, tables, combo, total_units, output_dir: Path) -> None:
    fractions = [i / total_units for i in range(total_units + 1)]
    enthalpies: List[float] = []
    for frac_a in fractions:
        frac_b = 1.0 - frac_a
        composition = [(combo[0], frac_a), (combo[1], frac_b)]
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        enthalpies.append(total_enthalpy)
    save_binary_plot(combo, fractions, enthalpies, output_dir)


def plot_ternary_combination(
    calculator,
    tables,
    combo,
    vectors: Sequence[Tuple[int, ...]],
    total_units: int,
    output_dir: Path,
) -> None:
    xs: List[float] = []
    ys: List[float] = []
    values: List[float] = []
    for vector in vectors:
        fractions = fractions_from_vector(vector, total_units)
        composition = list(zip(combo, fractions))
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
        x, y = barycentric_to_cartesian(fractions)
        xs.append(x)
        ys.append(y)
        values.append(total_enthalpy)
    save_ternary_plot(combo, xs, ys, values, output_dir)


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
            component_count = len(unique_elements)
            step_value = BINARY_STEP if component_count == 2 else TERNARY_STEP
            total_units, _ = normalize_step(step_value)
            combo = tuple(unique_elements)

            if component_count == 2:
                fractions = [i / total_units for i in range(total_units + 1)]
                enthalpies = []
                for frac_a in fractions:
                    frac_b = 1.0 - frac_a
                    composition = [(combo[0], frac_a), (combo[1], frac_b)]
                    total_enthalpy, _ = calculator.compute_multi_component_enthalpy(
                        tables, composition
                    )
                    enthalpies.append(total_enthalpy)
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
                    yaxis=dict(
                        title=dict(text=r"$\Delta H_{\mathrm{mix}}$ (kJ/mol)", font=PLOTLY_ELEMENT_FONT)
                    ),
                    template="plotly_white",
                )
                apply_plotly_base_style(fig)
                preview_and_maybe_save(fig, custom_dir / f"{combo[0]}-{combo[1]}.png")
            elif component_count == 3:
                vectors = build_fraction_vectors(3, total_units)
                a_vals = []
                b_vals = []
                c_vals = []
                enthalpies = []
                for vector in vectors:
                    fractions = fractions_from_vector(vector, total_units)
                    composition = list(zip(combo, fractions))
                    total_enthalpy, _ = calculator.compute_multi_component_enthalpy(
                        tables, composition
                    )
                    a_vals.append(fractions[0] * 100)
                    b_vals.append(fractions[1] * 100)
                    c_vals.append(fractions[2] * 100)
                    enthalpies.append(total_enthalpy)
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
                                title=dict(
                                    text=COLORBAR_LABEL_CONFIG.get("plotly_text"),
                                    font=PLOTLY_ELEMENT_FONT,
                                ),
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
                    template="plotly_white",
                )
                apply_plotly_base_style(fig)
                preview_and_maybe_save(fig, custom_dir / f"{combo[0]}-{combo[1]}-{combo[2]}.png")
            else:
                print("Quaternary plotting is not supported yet.")


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
    total_units, _ = normalize_step(step_value)
    combo = tuple(elements)

    if component_count == 2:
        fractions = [i / total_units for i in range(total_units + 1)]
        enthalpies: List[float] = []
        for frac_a in fractions:
            frac_b = 1.0 - frac_a
            composition = [(combo[0], frac_a), (combo[1], frac_b)]
            total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)
            enthalpies.append(total_enthalpy)
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
        return fig, f"{combo[0]}-{combo[1]}.png"

    if component_count == 3:
        vectors = build_fraction_vectors(3, total_units)
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
                        title=dict(
                            text=COLORBAR_LABEL_CONFIG.get("plotly_text"),
                            font=PLOTLY_ELEMENT_FONT,
                        ),
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
            template="plotly_white",
        )
        apply_plotly_base_style(fig)
        return fig, f"{combo[0]}-{combo[1]}-{combo[2]}.png"

    raise ValueError("Quaternary plotting is not supported yet.")


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
            fig.write_image(str(target))
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
        print("3) Batch quaternary (not supported)")
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
            print("Quaternary visualizations are not supported at the moment.")
        elif choice == "4":
            handle_custom_plot(calculator, tables, args.output_dir)
        elif choice in {"q", "quit", "exit"}:
            print("Bye.")
            break
        else:
            print("Invalid selection. Please choose 1–4 or q.")


if __name__ == "__main__":
    main()

"""Core utilities for enthalpy calculations and data loading."""

from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from enthalpy_config import DEFAULT_DATABASE_PATH, OMEGA_SHEETS, TETRA_VERTICES

Composition = List[Tuple[str, float]]

COMPOSITION_PATTERN = re.compile(
    r"([A-Za-z]{1,2})\s*(?:[:=])?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Supported elements (symbol -> atomic number). Order is used to select A/B.
ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1,
    "Be": 4,
    "B": 5,
    "C": 6,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "In": 49,
    "Sn": 50,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Sm": 62,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Pb": 82,
    "Th": 90,
}


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip()
    if not normalized:
        raise ValueError("Element symbol cannot be empty.")
    if len(normalized) == 1:
        return normalized.upper()
    return normalized[0].upper() + normalized[1:].lower()


def normalize_step(step: float) -> Tuple[int, float]:
    """Return total units and adjusted step so 1.0 divides evenly."""
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


def load_omega_tables(path: Path = DEFAULT_DATABASE_PATH) -> Dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    xls = pd.ExcelFile(path)
    tables: Dict[str, pd.DataFrame] = {}
    for sheet in OMEGA_SHEETS:
        df = xls.parse(sheet)
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Element"})
        df["Element"] = df["Element"].astype(str).str.strip()
        df = df.set_index("Element")
        df.columns = [str(col).strip() for col in df.columns]
        tables[sheet] = df

    elements = tables[OMEGA_SHEETS[0]].index.tolist()
    missing = [el for el in elements if el not in ATOMIC_NUMBERS]
    if missing:
        raise KeyError(
            "Atomic numbers are missing for the following elements: "
            + ", ".join(missing)
        )
    return tables


def list_supported_elements(tables: Dict[str, pd.DataFrame]) -> List[str]:
    """Return element list from the first Omega sheet."""
    first_sheet = OMEGA_SHEETS[0]
    return list(tables[first_sheet].index)


def ordered_pair(elem_a: str, elem_b: str) -> Tuple[str, str]:
    if elem_a not in ATOMIC_NUMBERS or elem_b not in ATOMIC_NUMBERS:
        raise KeyError("Element not defined in the database.")
    if elem_a == elem_b:
        raise ValueError("Two different elements are required.")

    if ATOMIC_NUMBERS[elem_a] <= ATOMIC_NUMBERS[elem_b]:
        return elem_a, elem_b
    return elem_b, elem_a


def lookup_omegas(
    tables: Dict[str, pd.DataFrame], elem_a: str, elem_b: str
) -> Tuple[List[float], Tuple[str, str]]:
    from_ordered, to_ordered = ordered_pair(elem_a, elem_b)
    omegas: List[float] = []
    for sheet in OMEGA_SHEETS:
        df = tables[sheet]
        try:
            value = df.at[from_ordered, to_ordered]
        except KeyError as exc:
            raise KeyError(
                f"Pair {from_ordered}-{to_ordered} is missing from sheet {sheet}."
            ) from exc
        if pd.isna(value):
            raise KeyError(
                f"Pair {from_ordered}-{to_ordered} has no value on sheet {sheet}."
            )
        omegas.append(float(value))
    return omegas, (from_ordered, to_ordered)


def parse_composition_line(line: str) -> Composition:
    matches = COMPOSITION_PATTERN.findall(line)
    if not matches:
        raise ValueError("Could not parse the input. Use format like 'Fe 20 Al 80'.")

    order: List[str] = []
    totals: Dict[str, float] = {}
    for symbol_raw, value_raw in matches:
        symbol = normalize_symbol(symbol_raw)
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"Element {symbol} is not supported.")
        value = float(value_raw)
        if value < 0:
            raise ValueError("Percentages must be non-negative.")
        if symbol not in totals:
            order.append(symbol)
            totals[symbol] = 0.0
        totals[symbol] += value

    if len(order) < 2:
        raise ValueError("At least two different elements are required.")

    total = sum(totals.values())
    if total <= 0:
        raise ValueError("The sum of the percentages must be positive.")

    return [(symbol, totals[symbol] / total) for symbol in order]


def compute_multi_component_enthalpy(
    tables: Dict[str, pd.DataFrame],
    composition: Composition,
) -> Tuple[float, List[Tuple[str, Tuple[float, float], float]]]:
    total_enthalpy = 0.0
    details: List[Tuple[str, Tuple[float, float], float]] = []

    for (sym_a, frac_a), (sym_b, frac_b) in combinations(composition, 2):
        omegas, ordered = lookup_omegas(tables, sym_a, sym_b)
        fraction_map = {sym_a: frac_a, sym_b: frac_b}
        ordered_fractions = [fraction_map[name] for name in ordered]
        c_a, c_b = ordered_fractions
        delta = c_a - c_b
        polynomial = sum(omega * (delta ** k) for k, omega in enumerate(omegas))
        delta_h = 4.0 * polynomial * c_a * c_b

        details.append((f"{ordered[0]}-{ordered[1]}", (c_a, c_b), delta_h))
        total_enthalpy += delta_h

    return total_enthalpy, details


def build_fraction_vectors(count: int, total_units: int) -> List[Tuple[int, ...]]:
    def recurse(index: int, remaining: int, current: List[int]):
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
    if total_units == 0:
        raise ValueError("Total units must be positive.")
    return tuple(value / total_units for value in vector)


def barycentric_to_cartesian(fractions: Sequence[float]) -> Tuple[float, float]:
    if len(fractions) != 3:
        raise ValueError("Ternary barycentric conversion requires exactly 3 fractions.")
    a, b, c = fractions
    x = b + 0.5 * c
    y = (3**0.5 / 2.0) * c
    return x, y


def build_binary_curve(calculator, tables, combo: Sequence[str], total_units: int):
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
):
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


def build_quaternary_points(
    calculator,
    tables,
    combo: Sequence[str],
    vectors: Sequence[Tuple[int, ...]],
    total_units: int,
):
    x_vals: List[float] = []
    y_vals: List[float] = []
    z_vals: List[float] = []
    enthalpies: List[float] = []
    fractions_list: List[Tuple[float, ...]] = []

    for vector in vectors:
        fractions = fractions_from_vector(vector, total_units)
        composition = list(zip(combo, fractions))
        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(tables, composition)

        # Barycentric embedding into regular tetrahedron
        x = sum(frac * vert[0] for frac, vert in zip(fractions, TETRA_VERTICES))
        y = sum(frac * vert[1] for frac, vert in zip(fractions, TETRA_VERTICES))
        z = sum(frac * vert[2] for frac, vert in zip(fractions, TETRA_VERTICES))

        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
        enthalpies.append(total_enthalpy)
        fractions_list.append(fractions)

    return x_vals, y_vals, z_vals, enthalpies, fractions_list

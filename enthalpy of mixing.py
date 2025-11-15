#!/usr/bin/env python3
"""
Binary alloy enthalpy of mixing calculator driven entirely by the Excel
line-input workflow: type a single line containing any number of element
symbols and their atomic percentages, and the script will look up Ω₀–Ω₃ from
the Excel matrices for every pair and sum their contributions.
"""

from __future__ import annotations

import re
import site
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _extend_sys_path_with_site_packages() -> None:
    """Ensure user/virtual-env site-packages folders are available."""
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

OMEGA_SHEETS = ("U0", "U1", "U2", "U3")
DEFAULT_DATABASE_PATH = Path("Data/Element pair data base matrices.xlsx")
# Pattern to parse "Element value" pairs (handles delimiters like :, =, or whitespace)
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

Composition = List[Tuple[str, float]]

@dataclass
class BinaryMixture:
    """Component names, mole fractions, and interaction coefficients."""

    names: List[str]
    mole_fractions: List[float]  # [c_A, c_B]
    omegas: List[float]  # Ω_0 ... Ω_3

    def enthalpy_of_mixing(self) -> float:
        """Compute ΔH_mix (kJ/mol)."""
        c_a, c_b = self.mole_fractions
        delta = c_a - c_b
        polynomial = sum(omega * (delta ** k) for k, omega in enumerate(self.omegas))
        return 4.0 * polynomial * c_a * c_b

    def polynomial_terms(self) -> List[float]:
        """Return Ω_k · (c_A - c_B)^k for each polynomial order."""
        c_a, c_b = self.mole_fractions
        delta = c_a - c_b
        return [omega * (delta ** k) for k, omega in enumerate(self.omegas)]


@dataclass(frozen=True)
class PairContribution:
    """Represents the contribution of a binary pair to the total enthalpy."""

    pair: Tuple[str, str]
    fractions: Tuple[float, float]
    delta_h: float


def normalize_symbol(symbol: str) -> str:
    """Return a capitalized element symbol."""
    normalized = symbol.strip()
    if not normalized:
        raise ValueError("Element symbol cannot be empty.")
    if len(normalized) == 1:
        return normalized.upper()
    return normalized[0].upper() + normalized[1:].lower()


def load_omega_tables(path: Path) -> Dict[str, pd.DataFrame]:
    """Load Ω matrices from the Excel workbook."""
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


def ordered_pair(elem_a: str, elem_b: str) -> Tuple[str, str]:
    """Return elements ordered by atomic number (A atomic number < B)."""
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
    """Fetch Ω coefficients for the ordered pair."""
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


def parse_composition_line(line: str) -> List[Tuple[str, float]]:
    """Parse a single line like 'Fe 20 Al 30 Ni 50' into normalized fractions."""
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
) -> Tuple[float, List[PairContribution]]:
    """Sum ΔH_mix contributions over every unique pair in the composition."""
    total_enthalpy = 0.0
    details: List[PairContribution] = []

    for (sym_a, frac_a), (sym_b, frac_b) in combinations(composition, 2):
        try:
            omegas, ordered = lookup_omegas(tables, sym_a, sym_b)
        except KeyError as exc:
            raise KeyError(f"Pair {sym_a}-{sym_b}: {exc}") from exc

        fraction_map = {sym_a: frac_a, sym_b: frac_b}
        ordered_fractions = [fraction_map[name] for name in ordered]
        mixture = BinaryMixture(list(ordered), ordered_fractions, omegas)
        delta_h = mixture.enthalpy_of_mixing()

        details.append(
            PairContribution(
                pair=tuple(ordered),
                fractions=tuple(ordered_fractions),
                delta_h=delta_h,
            )
        )
        total_enthalpy += delta_h

    return total_enthalpy, details


def prompt_excel_path(default_path: Path) -> Path:
    """Prompt the user for an Excel path and fall back to the default path."""
    raw = input(f"Excel workbook path (press Enter for {default_path}): ").strip()
    return Path(raw) if raw else default_path


def print_composition_summary(composition: Composition) -> None:
    """Display normalized atomic fractions."""
    print("\nNormalized atomic fractions:")
    for symbol, fraction in composition:
        print(f"{symbol:<5s} x = {fraction:.4f} ({fraction * 100:.2f}%)")


def print_pair_contributions(contributions: Sequence[PairContribution]) -> None:
    """Display the ΔH contribution for every binary pair."""
    if not contributions:
        print("\nNo valid element pairs were found.")
        return

    print("\nPairwise contributions (kJ/mol):")
    for contribution in contributions:
        element_a, element_b = contribution.pair
        c_a, c_b = contribution.fractions
        print(
            f"{element_a}-{element_b:<7s} ΔH = {contribution.delta_h:>10.5f} "
            f"(c_A={c_a:.4f}, c_B={c_b:.4f})"
        )


def print_total_enthalpy(total: float) -> None:
    """Display the total enthalpy of mixing for the alloy."""
    print(f"\nTotal ΔH_mix = {total:.5f} kJ/mol\n")


def run_excel_line_calculator() -> None:
    """Excel-driven calculator that accepts a single-line multi-element input."""
    print("\n--- Excel-driven multi-component calculator ---")
    path = prompt_excel_path(DEFAULT_DATABASE_PATH)

    try:
        tables = load_omega_tables(path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to load Excel data: {exc}")
        return

    print("Example input: Fe 20 Al 30 Ni 50  or  Fe=20,Al=30,Ni=50  (empty line to exit)")
    while True:
        try:
            line = input("Enter elements and atomic percentages: ").strip()
        except EOFError:
            print("\nEOF detected. Exiting Excel mode.\n")
            return
        if not line:
            print("Leaving Excel mode.\n")
            return

        try:
            composition = parse_composition_line(line)
        except ValueError as exc:
            print(f"Invalid input: {exc}")
            continue

        try:
            total_enthalpy, details = compute_multi_component_enthalpy(
                tables, composition
            )
        except KeyError as exc:
            print(exc)
            continue

        print_composition_summary(composition)
        print_pair_contributions(details)
        print_total_enthalpy(total_enthalpy)

def main() -> None:
    """Entry point; run the Excel calculator once."""
    try:
        run_excel_line_calculator()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Single-composition enthalpy of mixing calculator (interactive/CLI).
Relies on enthalpy_core for data loading and ΔH computations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from enthalpy_config import DEFAULT_DATABASE_PATH
from enthalpy_core import (
    compute_multi_component_enthalpy,
    list_supported_elements,
    load_omega_tables,
    normalize_symbol,
    parse_composition_line,
)


def print_composition_summary(composition: List[Tuple[str, float]]) -> None:
    print("\nNormalized atomic fractions:")
    for symbol, fraction in composition:
        print(f"{symbol:<5s} x = {fraction:.4f} ({fraction * 100:.2f}%)")


def print_pair_contributions(contributions: List[Tuple[str, Tuple[float, float], float]]) -> None:
    if not contributions:
        print("\nNo valid element pairs were found.")
        return
    print("\nPairwise contributions (kJ/mol):")
    for pair, (c_a, c_b), delta_h in contributions:
        print(f"{pair:<10s} ΔH = {delta_h:>10.5f} (c_A={c_a:.4f}, c_B={c_b:.4f})")


def print_total_enthalpy(total: float) -> None:
    print(f"\nTotal ΔH_mix = {total:.5f} kJ/mol\n")


def run_interactive(tables) -> None:
    print("Example input: Fe 20 Al 30 Ni 50  or  Fe=20,Al=30,Ni=50  (empty line to exit)")
    while True:
        try:
            line = input("Enter elements and atomic percentages: ").strip()
        except EOFError:
            print("\nEOF detected. Exiting.\n")
            return
        if not line:
            print("Leaving interactive mode.\n")
            return

        try:
            composition = parse_composition_line(line)
        except ValueError as exc:
            print(f"Invalid input: {exc}")
            continue

        # normalize symbols via enthalpy_core? parse already normalizes
        composition = [(normalize_symbol(sym), frac) for sym, frac in composition]

        try:
            total_enthalpy, details = compute_multi_component_enthalpy(tables, composition)
        except KeyError as exc:
            print(exc)
            continue

        print_composition_summary(composition)
        print_pair_contributions(details)
        print_total_enthalpy(total_enthalpy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary alloy enthalpy of mixing calculator (Excel-driven)."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"Path to the Omega matrices workbook (default: {DEFAULT_DATABASE_PATH}).",
    )
    parser.add_argument(
        "--line",
        help='Optional composition line to evaluate once (e.g., "Fe 20 Al 80").',
    )
    parser.add_argument(
        "--list-elements",
        action="store_true",
        help="Print supported elements from the workbook and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        tables = load_omega_tables(args.excel)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to load Excel data: {exc}")
        return

    if args.list_elements:
        print("Supported elements:")
        for el in list_supported_elements(tables):
            print(f"- {el}")
        return

    if args.line:
        try:
            composition = parse_composition_line(args.line)
            composition = [(normalize_symbol(sym), frac) for sym, frac in composition]
        except ValueError as exc:
            print(f"Invalid input: {exc}")
            return
        try:
            total_enthalpy, details = compute_multi_component_enthalpy(tables, composition)
        except KeyError as exc:
            print(exc)
            return

        print_composition_summary(composition)
        print_pair_contributions(details)
        print_total_enthalpy(total_enthalpy)
        return

    try:
        run_interactive(tables)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch enthalpy of mixing calculator.

Iterates over all element combinations (size 2–4 by default) defined in the Excel
data source, enumerates compositions using a configurable step (default 0.1) and
minimum fraction, evaluates the enthalpy of mixing via the interactive calculator
module, and streams the aggregated results to an Excel workbook with individual
sheets for binary, ternary, and quaternary alloys.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import pandas as pd

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


@dataclass(frozen=True)
class CompositionVector:
    """Represents a single composition (element symbols + mole fractions)."""

    elements: Tuple[str, ...]
    fractions: Tuple[float, ...]


def build_fraction_vectors(
    count: int,
    total_units: int,
    min_units: int,
) -> List[Tuple[int, ...]]:
    """Enumerate integer solutions that sum to total_units with per-element minimum."""

    def recurse(index: int, remaining: int, current: List[int]) -> Iterator[Tuple[int, ...]]:
        if index == count - 1:
            if remaining >= min_units:
                yield tuple(current + [remaining])
            return
        max_units = remaining - min_units * (count - index - 1)
        for value in range(min_units, max_units + 1):
            current.append(value)
            yield from recurse(index + 1, remaining - value, current)
            current.pop()

    if min_units * count > total_units:
        return []
    return list(recurse(0, total_units, []))


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


def generate_compositions(
    elements: Sequence[str],
    fraction_vectors: Sequence[Tuple[int, ...]],
    total_units: int,
) -> Iterable[CompositionVector]:
    """Convert unit-based vectors into floating-point mole fractions."""
    for vector in fraction_vectors:
        fractions = tuple(unit / total_units for unit in vector)
        yield CompositionVector(tuple(elements), fractions)


# --------------------------------------------------------------------------- #
# Excel streaming helper
# --------------------------------------------------------------------------- #


class SheetWriter:
    """Streaming writer that appends rows to an Excel worksheet."""

    def __init__(self, writer: pd.ExcelWriter, sheet_name: str, columns: List[str]) -> None:
        self.writer = writer
        self.sheet_name = sheet_name
        self.columns = columns
        self.row_offset = 0
        self.header_written = False

    def write_rows(self, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        df = df.reindex(columns=self.columns)
        df.to_excel(
            self.writer,
            sheet_name=self.sheet_name,
            startrow=self.row_offset,
            index=False,
            header=not self.header_written,
        )
        if not self.header_written:
            self.header_written = True
            self.row_offset += len(df) + 1  # include header row
        else:
            self.row_offset += len(df)

    def finalize(self) -> None:
        """Ensure the sheet exists even if no data was written."""
        if not self.header_written:
            empty = pd.DataFrame(columns=self.columns)
            empty.to_excel(
                self.writer,
                sheet_name=self.sheet_name,
                index=False,
                header=True,
            )
            self.header_written = True
            self.row_offset = 1


# --------------------------------------------------------------------------- #
# Core batch processing
# --------------------------------------------------------------------------- #


def sheet_columns(component_count: int) -> List[str]:
    """Generate column names for a given component count."""
    columns = ["Elements"]
    for idx in range(1, component_count + 1):
        columns.extend([f"Element{idx}", f"x{idx}", f"x{idx}_pct"])
    columns.append("DeltaH_kJ_per_mol")
    return columns


def build_row(
    elements: Sequence[str],
    fractions: Sequence[float],
    delta_h: float,
) -> Dict[str, object]:
    """Format a row dict for DataFrame/appending."""
    row: Dict[str, object] = {"Elements": "-".join(elements), "DeltaH_kJ_per_mol": delta_h}
    for idx, (symbol, fraction) in enumerate(zip(elements, fractions), start=1):
        row[f"Element{idx}"] = symbol
        row[f"x{idx}"] = round(fraction, 6)
        row[f"x{idx}_pct"] = round(fraction * 100.0, 4)
    return row


def iterate_combinations(
    elements: Sequence[str],
    size: int,
) -> Iterable[Tuple[str, ...]]:
    """Wrapper around itertools.combinations for readability."""
    return combinations(elements, size)


def run_batch(
    calculator,
    excel_path: Path,
    min_components: int,
    max_components: int,
    elements: Sequence[str],
    step: float,
    min_fraction: float,
    chunk_size: int,
    output_path: Path,
) -> None:
    """Main batch-processing routine."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if min_components > max_components:
        raise ValueError("min_components cannot exceed max_components.")
    if not (0 < step <= 1):
        raise ValueError("Step must be within (0, 1].")
    if not (0 < min_fraction <= 1):
        raise ValueError("min_fraction must be within (0, 1].")

    tables = calculator.load_omega_tables(excel_path)
    available_elements = [el for el in tables[calculator.OMEGA_SHEETS[0]].index if el in elements]
    if not available_elements:
        raise ValueError("No valid elements were selected for processing.")

    total_units, actual_step = normalize_step(step)
    min_units = max(1, round(min_fraction * total_units))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for component_count in range(min_components, max_components + 1):
            sheet_name = f"{component_count}-component"
            columns = sheet_columns(component_count)
            sheet_writer = SheetWriter(writer, sheet_name, columns)
            if min_units * component_count > total_units:
                print(
                    f"[info] Skipping {component_count}-component alloys: "
                    f"min_fraction * count exceeds 100%.",
                    file=sys.stderr,
                )
                sheet_writer.finalize()
                continue

            fraction_vectors = build_fraction_vectors(component_count, total_units, min_units)
            if not fraction_vectors:
                print(
                    f"[info] No feasible compositions for {component_count} components.",
                    file=sys.stderr,
                )
                sheet_writer.finalize()
                continue

            combo_iter = iterate_combinations(available_elements, component_count)
            buffer: List[Dict[str, object]] = []
            skipped_combos = 0
            processed = 0

            for combo in combo_iter:
                combo_failed = False
                for composition in generate_compositions(combo, fraction_vectors, total_units):
                    comp_pairs = list(composition.elements)
                    comp_fractions = list(composition.fractions)
                    comp_zip = list(zip(comp_pairs, comp_fractions))
                    try:
                        total_enthalpy, _ = calculator.compute_multi_component_enthalpy(
                            tables, comp_zip
                        )
                    except KeyError:
                        combo_failed = True
                        break
                    buffer.append(build_row(comp_pairs, comp_fractions, total_enthalpy))
                    if len(buffer) >= chunk_size:
                        sheet_writer.write_rows(buffer)
                        buffer.clear()
                if combo_failed:
                    skipped_combos += 1
                else:
                    processed += 1

            if buffer:
                sheet_writer.write_rows(buffer)
                buffer.clear()
            sheet_writer.finalize()

            print(
                f"[summary] {component_count}-component alloys: "
                f"{processed} combinations processed, {skipped_combos} skipped.",
                file=sys.stderr,
            )


# --------------------------------------------------------------------------- #
# Argument parsing & entry point
# --------------------------------------------------------------------------- #


MENU_OPTIONS = {
    "1": ("Binary (2 components)", 2),
    "2": ("Ternary (3 components)", 3),
    "3": ("Quaternary (4 components)", 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch enthalpy calculator (2–4 components).")
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
        "--component-option",
        choices=MENU_OPTIONS.keys(),
        help="Preset options: 1=Binary, 2=Ternary, 3=Quaternary. "
        "If omitted, an interactive menu will appear.",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        help="Optional whitelist of element symbols (defaults to all available).",
    )
    parser.add_argument("--step", type=float, default=0.1, help="Mole fraction step (0 < step ≤ 1).")
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=None,
        help="Minimum fraction per element (defaults to the step size).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows to buffer before writing to Excel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Excel filename (defaults to Data/batch_enthalpy_results.xlsx).",
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
    min_fraction = args.min_fraction if args.min_fraction is not None else args.step

    element_pool = (
        [calculator.normalize_symbol(sym) for sym in args.elements]
        if args.elements
        else calculator.ATOMIC_NUMBERS.keys()
    )

    if args.component_option:
        _, component_count = MENU_OPTIONS[args.component_option]
    else:
        print("Select alloy type:")
        for key, (label, _) in MENU_OPTIONS.items():
            print(f"{key}) {label}")
        while True:
            choice = input("Enter option (1/2/3): ").strip()
            if choice in MENU_OPTIONS:
                _, component_count = MENU_OPTIONS[choice]
                break
            print("Invalid option. Please enter 1, 2, or 3.")

    try:
        run_batch(
            calculator=calculator,
            excel_path=excel_path,
            min_components=component_count,
            max_components=component_count,
            elements=list(element_pool),
            step=args.step,
            min_fraction=min_fraction,
            chunk_size=args.chunk_size,
            output_path=Path(args.output)
            if args.output
            else Path("Data/batch_enthalpy_results.xlsx"),
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Batch processing failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

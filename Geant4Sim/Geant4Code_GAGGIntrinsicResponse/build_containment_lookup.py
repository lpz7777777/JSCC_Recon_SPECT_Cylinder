#!/usr/bin/env python3
"""Combine four Geant4 intrinsic-response CSV files into a compact lookup."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


CATEGORIES = (
    ("first_pe", "first_pe_contained"),
    ("first_compton_second_pe", "first_compton_second_pe_contained"),
    ("first_compton_eventual_pe", "first_compton_eventual_pe_contained"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("gagg_intrinsic_containment_lookup.csv")
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow fewer than the four production size/energy combinations",
    )
    return parser.parse_args()


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one data row, got {len(rows)}")
    return rows[0]


def main() -> None:
    args = parse_args()
    rows = [read_single_row(path) for path in args.inputs]
    keys = {
        (float(row["width_mm"]), float(row["thickness_mm"]),
         float(row["height_mm"]), float(row["energy_keV"]))
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("duplicate crystal-size/energy input")
    required_keys = {
        (3.0, 3.0, 3.0, 218.0),
        (3.0, 3.0, 3.0, 440.0),
        (2.0, 6.0, 2.0, 218.0),
        (2.0, 6.0, 2.0, 440.0),
    }
    if not args.allow_partial and keys != required_keys:
        missing = sorted(required_keys - keys)
        extra = sorted(keys - required_keys)
        raise ValueError(f"production lookup mismatch: missing={missing}, extra={extra}")

    output_rows: list[dict[str, object]] = []
    for row in rows:
        events = int(row["events"])
        entered = int(row["entered"])
        first_compton = int(row["first_compton"])
        second_pe = int(row["first_compton_second_pe"])
        eventual_pe = int(row["first_compton_eventual_pe"])
        if entered != events:
            raise ValueError(
                f"not every primary entered the crystal: entered={entered}, events={events}"
            )
        if not 0 <= second_pe <= eventual_pe <= first_compton:
            raise ValueError("inconsistent first-Compton history counts")
        for total_name, contained_name in CATEGORIES:
            total = int(row[total_name])
            contained = int(row[contained_name])
            if not 0 <= contained <= total:
                raise ValueError(
                    f"invalid containment counts for {total_name}: {contained}/{total}"
                )
            probability = contained / total if total else math.nan
            standard_error = (
                math.sqrt(probability * (1.0 - probability) / total)
                if total and math.isfinite(probability)
                else math.nan
            )
            output_rows.append(
                {
                    "width_mm": row["width_mm"],
                    "thickness_mm": row["thickness_mm"],
                    "height_mm": row["height_mm"],
                    "energy_keV": row["energy_keV"],
                    "category": total_name,
                    "total": total,
                    "contained": contained,
                    "containment_probability": f"{probability:.12g}",
                    "binomial_standard_error": f"{standard_error:.12g}",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda item: (
            float(item["energy_keV"]), float(item["width_mm"]), str(item["category"])
        )))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

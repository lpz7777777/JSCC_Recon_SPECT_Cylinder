#!/usr/bin/env python3
"""Download NIST XCOM partial coefficients and generate CUDA lookup data."""

from __future__ import annotations

import csv
import html.parser
import math
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
CSV_PATH = ROOT / "nist_xcom_materials_1_1000keV.csv"
HEADER_PATH = ROOT / "nist_xcom_materials_1_1000keV.h"
ENERGIES_KEV = tuple(range(1, 1001))
# Complex compounds add constituent absorption-edge energies internally. Keeping
# user batches at 50 stays below XCOM's effective 100-energy response limit.
CHUNK_SIZE = 50

MATERIALS = (
    {"name": "NaI", "formula": "NaI", "density": 3.67, "endpoint": "xcom3_2"},
    {"name": "GAGG", "formula": "Gd3Al2Ga3O12", "density": 6.63, "endpoint": "xcom3_2"},
    {"name": "Pb", "symbol": "Pb", "density": 11.35, "endpoint": "xcom3_1"},
    {"name": "W", "symbol": "W", "density": 19.30, "endpoint": "xcom3_1"},
)


class TableParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() == "td" and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "td" and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag.lower() == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def request_chunk(material: dict[str, object], energies_keV: tuple[int, ...]) -> dict[int, tuple[float, float]]:
    values: dict[str, str] = {
        "Graph2": "on",
        "Graph3": "on",
        "Graph7": "on",
        "NumAdd": "1",
        "WindowXmin": "0.001",
        "WindowXmax": "1",
        "ResizeFlag": "on",
        "Energies": "\n".join(f"{energy / 1000.0:.6f}" for energy in energies_keV),
    }
    if material["endpoint"] == "xcom3_2":
        values["Formula"] = str(material["formula"])
        values["Name"] = str(material["name"])
    else:
        values["ZSym"] = str(material["symbol"])
        values["OutOpt"] = "PIC"

    request = urllib.request.Request(
        f"https://physics.nist.gov/cgi-bin/Xcom/{material['endpoint']}",
        data=urllib.parse.urlencode(values).encode("ascii"),
        headers={"User-Agent": "SPECT-system-matrix-XCOM-downloader/1.0"},
    )
    body = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("ascii", errors="replace")
            break
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))

    if "XCOM: Error" in body:
        raise RuntimeError(f"NIST XCOM rejected {material['name']} request")

    parser = TableParser()
    parser.feed(body)
    result: dict[int, tuple[float, float]] = {}
    requested = set(energies_keV)
    for row in parser.rows:
        if len(row) != 9:
            continue
        try:
            energy_keV = int(round(float(row[1]) * 1000.0))
            incoherent_cm2_g = float(row[3])
            photoelectric_cm2_g = float(row[4])
        except ValueError:
            continue
        if energy_keV in requested:
            result[energy_keV] = (photoelectric_cm2_g, incoherent_cm2_g)

    missing = requested.difference(result)
    if missing:
        raise RuntimeError(f"NIST XCOM response for {material['name']} missed {sorted(missing)}")
    return result


def download() -> dict[str, dict[int, tuple[float, float]]]:
    all_data: dict[str, dict[int, tuple[float, float]]] = {}
    for material in MATERIALS:
        name = str(material["name"])
        rows: dict[int, tuple[float, float]] = {}
        for start in range(0, len(ENERGIES_KEV), CHUNK_SIZE):
            chunk = ENERGIES_KEV[start : start + CHUNK_SIZE]
            rows.update(request_chunk(material, chunk))
            time.sleep(0.1)
        all_data[name] = rows
        print(f"Downloaded {len(rows)} energies for {name}")
    return all_data


def write_csv(data: dict[str, dict[int, tuple[float, float]]]) -> None:
    fields = ["energy_keV"]
    for material in MATERIALS:
        name = str(material["name"])
        fields.extend(
            (
                f"{name}_photoelectric_cm2_g",
                f"{name}_incoherent_cm2_g",
                f"{name}_mu_photoelectric_per_mm",
                f"{name}_mu_compton_per_mm",
            )
        )

    with CSV_PATH.open("w", newline="", encoding="ascii") as stream:
        stream.write("# Source: NIST XCOM Photon Cross Sections Database\n")
        stream.write("# URL: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html\n")
        stream.write("# Linear coefficient = mass interaction coefficient * density / 10 (1/mm)\n")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for energy in ENERGIES_KEV:
            row: dict[str, float | int] = {"energy_keV": energy}
            for material in MATERIALS:
                name = str(material["name"])
                density = float(material["density"])
                photoelectric, incoherent = data[name][energy]
                row[f"{name}_photoelectric_cm2_g"] = photoelectric
                row[f"{name}_incoherent_cm2_g"] = incoherent
                row[f"{name}_mu_photoelectric_per_mm"] = photoelectric * density / 10.0
                row[f"{name}_mu_compton_per_mm"] = incoherent * density / 10.0
            writer.writerow(row)


def float_literal(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite coefficient")
    return f"{value:.9e}f"


def write_header(data: dict[str, dict[int, tuple[float, float]]]) -> None:
    pe_values: list[float] = []
    compton_values: list[float] = []
    for material in MATERIALS:
        name = str(material["name"])
        density = float(material["density"])
        for energy in ENERGIES_KEV:
            photoelectric, incoherent = data[name][energy]
            pe_values.append(photoelectric * density / 10.0)
            compton_values.append(incoherent * density / 10.0)

    def format_array(values: list[float]) -> str:
        lines = []
        for start in range(0, len(values), 6):
            lines.append("    " + ", ".join(float_literal(value) for value in values[start : start + 6]))
        return ",\n".join(lines)

    text = f"""#pragma once

// Generated by download_nist_xcom.py. Do not hand-edit.
// NIST XCOM partial mass interaction coefficients, converted to linear 1/mm.
// Material order: NaI, GAGG (Gd3Al2Ga3O12), Pb, W.

constexpr int kXcomMaterialCount = {len(MATERIALS)};
constexpr int kXcomEnergyMinKeV = {ENERGIES_KEV[0]};
constexpr int kXcomEnergyMaxKeV = {ENERGIES_KEV[-1]};
constexpr int kXcomEnergyCount = {len(ENERGIES_KEV)};

enum XcomMaterialId {{
    kMaterialNaI = 0,
    kMaterialGAGG = 1,
    kMaterialPb = 2,
    kMaterialW = 3,
    kMaterialVacuum = -1,
}};

static const float kXcomMuPhotoelectric[kXcomMaterialCount * kXcomEnergyCount] = {{
{format_array(pe_values)}
}};

static const float kXcomMuCompton[kXcomMaterialCount * kXcomEnergyCount] = {{
{format_array(compton_values)}
}};
"""
    HEADER_PATH.write_text(text, encoding="ascii")


def main() -> None:
    data = download()
    write_csv(data)
    write_header(data)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {HEADER_PATH}")


if __name__ == "__main__":
    main()

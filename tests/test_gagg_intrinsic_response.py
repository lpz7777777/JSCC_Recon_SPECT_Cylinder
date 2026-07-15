import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT = REPO_ROOT / "Geant4Sim" / "Geant4Code_GAGGIntrinsicResponse"
SOURCE = PROJECT / "gagg_intrinsic.cc"
LOOKUP_SCRIPT = PROJECT / "build_containment_lookup.py"


class GaggIntrinsicSourceTests(unittest.TestCase):
    def test_energy_deposit_from_all_tracks_precedes_primary_history_filter(self):
        source = SOURCE.read_text(encoding="utf-8")
        add_deposit = source.index("AddEnergyDeposit(step->GetTotalEnergyDeposit())")
        primary_filter = source.index("track->GetTrackID() != 1", add_deposit)
        self.assertLess(add_deposit, primary_filter)

    def test_production_macros_cover_required_size_energy_pairs(self):
        observed = set()
        for path in sorted((PROJECT / "macros").glob("GAGG_*.mac")):
            values = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) >= 2 and (
                    fields[0].startswith("/study/") or fields[0] == "/run/beamOn"
                ):
                    values[fields[0]] = fields[1]
            observed.add(
                (
                    float(values["/study/crystalWidth"]),
                    float(values["/study/crystalThickness"]),
                    float(values["/study/crystalHeight"]),
                    float(values["/study/energy"]),
                )
            )
            self.assertEqual(values["/run/beamOn"], "10000000")
            self.assertNotIn("/run/numberOfThreads", path.read_text(encoding="utf-8"))
        self.assertEqual(
            observed,
            {
                (3.0, 3.0, 3.0, 218.0),
                (3.0, 3.0, 3.0, 440.0),
                (2.0, 6.0, 2.0, 218.0),
                (2.0, 6.0, 2.0, 440.0),
            },
        )


class GaggLookupBuilderTests(unittest.TestCase):
    FIELDNAMES = [
        "width_mm",
        "thickness_mm",
        "height_mm",
        "energy_keV",
        "tolerance_eV",
        "events",
        "entered",
        "full_energy_all",
        "first_pe",
        "first_pe_contained",
        "first_pe_containment",
        "first_compton",
        "first_compton_second_pe",
        "first_compton_second_pe_contained",
        "first_compton_second_pe_containment",
        "first_compton_eventual_pe",
        "first_compton_eventual_pe_contained",
        "first_compton_eventual_pe_containment",
        "first_other",
        "no_interaction",
    ]

    def write_input(self, path: Path, energy: int) -> None:
        row = {
            "width_mm": 3,
            "thickness_mm": 3,
            "height_mm": 3,
            "energy_keV": energy,
            "tolerance_eV": 1,
            "events": 1000,
            "entered": 1000,
            "full_energy_all": 600,
            "first_pe": 500,
            "first_pe_contained": 400,
            "first_pe_containment": 0.8,
            "first_compton": 300,
            "first_compton_second_pe": 200,
            "first_compton_second_pe_contained": 100,
            "first_compton_second_pe_containment": 0.5,
            "first_compton_eventual_pe": 250,
            "first_compton_eventual_pe_contained": 150,
            "first_compton_eventual_pe_containment": 0.6,
            "first_other": 100,
            "no_interaction": 100,
        }
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerow(row)

    def test_lookup_contains_component_specific_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.csv"
            second = root / "second.csv"
            output = root / "lookup.csv"
            self.write_input(first, 218)
            self.write_input(second, 440)
            subprocess.run(
                [
                    sys.executable,
                    str(LOOKUP_SCRIPT),
                    str(first),
                    str(second),
                    "--output",
                    str(output),
                    "--allow-partial",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        probabilities = {
            (row["energy_keV"], row["category"]): float(
                row["containment_probability"]
            )
            for row in rows
        }
        self.assertEqual(probabilities[("218", "first_pe")], 0.8)
        self.assertEqual(
            probabilities[("440", "first_compton_second_pe")], 0.5
        )
        for row in rows:
            self.assertGreater(float(row["binomial_standard_error"]), 0.0)


if __name__ == "__main__":
    unittest.main()

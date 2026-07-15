import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_ACTION = REPO_ROOT / "Geant4Sim" / "Geant4Code" / "src" / "EventAction.cc"
EHE_CODE = REPO_ROOT / "Geant4Sim" / "Geant4Code_EHE"
JSCC_CODE = REPO_ROOT / "Geant4Sim" / "Geant4Code"
RESPONSE_STUDY_CODE = (
    REPO_ROOT / "Geant4Sim" / "Geant4Code_CntStatResponseStudy"
)


class Geant4StepVolumeAttributionSourceTests(unittest.TestCase):
    def test_production_projects_use_pre_step_volume_for_energy_deposit(self):
        for code_dir in [JSCC_CODE, EHE_CODE]:
            with self.subTest(project=code_dir.name):
                source = (code_dir / "src" / "SteppingAction.cc").read_text(
                    encoding="utf-8"
                )
                self.assertIn("GetTotalEnergyDeposit()", source)
                self.assertIn("GetPreStepPoint()", source)
                volume_lookup_prefix = source[: source.index("GetTotalEnergyDeposit()") + 500]
                self.assertNotIn("GetPostStepPoint()->GetTouchableHandle()", volume_lookup_prefix)


class Geant4EventClassificationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = EVENT_ACTION.read_text(encoding="utf-8")
        match = re.search(
            r"void\s+EventAction::EndOfEventAction\s*\([^)]*\)\s*\{(?P<body>.*)\n\}",
            cls.source,
            flags=re.DOTALL,
        )
        if match is None:
            raise AssertionError("Cannot locate EventAction::EndOfEventAction.")
        cls.body = match.group("body")

    def test_cntstat_scans_all_crystals(self):
        self.assertIn("for(int i=0; i<nScinNum; i++)", self.body)
        self.assertIn("run->AddCnt218(i);", self.body)
        self.assertIn("run->AddCnt440(i);", self.body)
        self.assertNotIn("maxIdx", self.body)
        self.assertNotIn("maxE", self.body)

    def test_cntstat_does_not_short_circuit_compton_classification(self):
        self.assertNotIn("return;", self.body)
        cntstat_end = max(
            self.body.index("run->AddCnt218(i);"),
            self.body.index("run->AddCnt440(i);"),
        )
        list_classification = self.body.index("if (Flag2 != -1")
        add_list = self.body.index("run->AddList(")
        self.assertLess(cntstat_end, list_classification)
        self.assertLess(list_classification, add_list)

    def test_both_energy_windows_are_checked_independently(self):
        window_checks = re.findall(
            r"if\(energy >= fWin(218|440)_lo && energy <= fWin\1_hi\)", self.body
        )
        self.assertCountEqual(window_checks, ["218", "440"])


class EHECntStatOnlySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.event_source = (EHE_CODE / "src" / "EventAction.cc").read_text(
            encoding="utf-8"
        )
        cls.all_classification_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [
                EHE_CODE / "src" / "EventAction.cc",
                EHE_CODE / "include" / "EventAction.hh",
                EHE_CODE / "src" / "SteppingAction.cc",
                EHE_CODE / "src" / "Run.cc",
                EHE_CODE / "include" / "Run.hh",
                EHE_CODE / "src" / "RunAction.cc",
            ]
        )

    def test_ehe_cntstat_scans_all_detector_bins(self):
        self.assertIn("for(int i=0; i<nScinNum; i++)", self.event_source)
        self.assertIn("run->AddCnt218(i);", self.event_source)
        self.assertIn("run->AddCnt440(i);", self.event_source)
        self.assertNotIn("maxIdx", self.event_source)
        self.assertNotIn("maxE", self.event_source)
        self.assertNotIn("return;", self.event_source)

    def test_ehe_has_no_compton_or_list_path(self):
        forbidden = [
            "AddList",
            "GetList",
            "GetTotalCount",
            "List.csv",
            "NumCompt",
            "Flag_Compt",
            'GetProcessName() == "compt"',
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.all_classification_sources)


class CntStatResponseStudySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.event_source = (RESPONSE_STUDY_CODE / "src" / "EventAction.cc").read_text(
            encoding="utf-8"
        )
        cls.run_header = (RESPONSE_STUDY_CODE / "include" / "Run.hh").read_text(
            encoding="utf-8"
        )
        cls.run_source = (RESPONSE_STUDY_CODE / "src" / "Run.cc").read_text(
            encoding="utf-8"
        )
        cls.run_action = (RESPONSE_STUDY_CODE / "src" / "RunAction.cc").read_text(
            encoding="utf-8"
        )

    def test_primary_energy_is_read_once_per_event(self):
        self.assertIn("GetPrimaryVertex(0)", self.event_source)
        self.assertIn("GetPrimary(0)", self.event_source)
        self.assertIn("GetKineticEnergy()", self.event_source)
        self.assertIn("AddPrimary218()", self.event_source)
        self.assertIn("AddPrimary440()", self.event_source)

    def test_requested_source_separated_cntstats_are_filled(self):
        self.assertIn("for(int i=0; i<nScinNum; i++)", self.event_source)
        for method in [
            "AddCnt218From218(i)",
            "AddCnt218From440(i)",
            "AddCnt440From440(i)",
        ]:
            with self.subTest(method=method):
                self.assertIn(method, self.event_source)

    def test_component_arrays_and_primary_counts_are_merged(self):
        for token in [
            "LocalCnt218From218",
            "LocalCnt218From440",
            "LocalCnt440From440",
            "PrimaryCount218",
            "PrimaryCount440",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, self.run_header)
                self.assertIn(token, self.run_source)
        self.assertIn("localRun->PrimaryCount218", self.run_source)
        self.assertIn("localRun->PrimaryCount440", self.run_source)

    def test_requested_output_filenames_are_exact(self):
        for filename in [
            "CntStat218_From218.csv",
            "CntStat218_From440.csv",
            "CntStat440_From440.csv",
            "PrimaryCount218.csv",
            "PrimaryCount440.csv",
        ]:
            with self.subTest(filename=filename):
                self.assertIn(f'"{filename}"', self.run_action)


class EHEGeometryPlacementSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = (EHE_CODE / "include" / "DetectorConstruction.hh").read_text(
            encoding="utf-8"
        )
        cls.params_readme = (
            REPO_ROOT
            / "Auxiliary_Studies"
            / "GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
            / "runs"
            / "EHE_PbNaI_218keV"
            / "Params_README.txt"
        ).read_text(encoding="utf-8")

    def test_ehe_front_face_matches_jscc_and_generated_params(self):
        self.assertRegex(
            self.header,
            r"kCommonFrontFaceDistance\s*=\s*198\.5\s*;",
        )
        self.assertRegex(
            self.header,
            r"kFovToCollimatorOrigin\s*=\s*\n?\s*"
            r"kCommonFrontFaceDistance\s*\+\s*kCollimatorThicknessY\s*/\s*2\.0\s*;",
        )
        self.assertIn(
            "shared_JSCC_detector_and_EHE_collimator_front_face_mm = 198.5",
            self.params_readme,
        )
        self.assertIn(
            "cuda_fov_to_local_y_origin_mm = 223.75",
            self.params_readme,
        )


if __name__ == "__main__":
    unittest.main()

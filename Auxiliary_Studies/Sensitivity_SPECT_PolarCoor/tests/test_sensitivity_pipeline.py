from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from spect_sensitivity import (
    ComptonPhysicsConfig,
    SensitivityRunConfig,
    run_sensitivity_calculation,
)
from spect_sensitivity.kernel import BatchDiagnostics
from spect_sensitivity.pipeline import _load_checkpoint, _save_checkpoint
import spect_sensitivity.pipeline as sensitivity_pipeline


class SensitivityPipelineTest(unittest.TestCase):
    def _create_fixture(self, root: Path, density_basis: bool = True) -> tuple[Path, Path]:
        factor_dir = root / "511keV_RotateNum2"
        factor_dir.mkdir()
        detector = np.asarray(
            [
                [1, 0.0, 10.0, 0.0],
                [2, -2.0, 12.0, 1.0],
                [3, 2.0, 14.0, -1.0],
                [4, 8.0, 20.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.savetxt(
            factor_dir / "Detector.csv",
            detector,
            delimiter=",",
            header="index,x,y,z",
            comments="",
        )

        axis = np.linspace(-8.0, 8.0, 8, dtype=np.float32)
        coordinates = np.asarray(
            [[x, y, 0.0] for y in axis for x in axis], dtype=np.float32
        )
        np.savetxt(factor_dir / "coor_polar_full.csv", coordinates, delimiter=",")
        pixel_count = coordinates.shape[0]

        system_matrix = np.ones((4, pixel_count), dtype=np.float32)
        system_matrix[0] += np.linspace(0.0, 0.5, pixel_count, dtype=np.float32)
        system_matrix.T.tofile(factor_dir / "SysMat_polar")
        if density_basis:
            volumes = np.full(pixel_count, 2.5, dtype=np.float64)
            volumes.tofile(factor_dir / "polar_cell_volume_mm3.float64")
            (factor_dir / "factor_manifest.json").write_text(
                json.dumps({"maps_activity_density": True}), encoding="utf-8"
            )

        rotation = np.column_stack(
            [
                np.arange(1, pixel_count + 1, dtype=np.int64),
                np.roll(np.arange(1, pixel_count + 1, dtype=np.int64), 1),
            ]
        )
        np.savetxt(factor_dir / "RotMat_full.csv", rotation, delimiter=",", fmt="%d")

        list_path = root / "compton.csv"
        events = np.tile(np.asarray([[1, 0.20, 4, 0.311, 1]], dtype=np.float32), (12, 1))
        np.savetxt(list_path, events, delimiter=",")
        return factor_dir, list_path

    def test_pipeline_normalization_and_rotation_average(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            factor_dir, list_path = self._create_fixture(root)
            output_dir = root / "result"
            config = SensitivityRunConfig(
                factor_dir=factor_dir,
                compton_paths=(list_path,),
                output_dir=output_dir,
                source_photons=1000.0,
                physics=ComptonPhysicsConfig(
                    energy_mev=0.511,
                    energy_threshold_sum_mev=0.46,
                    min_event_effective_support=1.0,
                ),
                system_matrix_path=factor_dir / "SysMat_polar",
                detector_path=factor_dir / "Detector.csv",
                coordinate_path=factor_dir / "coor_polar_full.csv",
                rotation_path=factor_dir / "RotMat_full.csv",
                rotate_num=2,
                event_fraction=0.5,
                batch_size=3,
                device="cpu",
                seed=7,
                expected_detector_count=4,
                checkpoint_every_batches=1,
                progress_every_batches=1,
            )
            metadata = run_sensitivity_calculation(config)

            sensitivity = np.fromfile(output_dir / "Sensi_d", dtype=np.float32)
            raw_sensitivity = np.fromfile(output_dir / "Sensi_d_raw", dtype=np.float32)
            self.assertEqual(sensitivity.size, 64)
            self.assertEqual(raw_sensitivity.size, 64)
            self.assertTrue(np.isfinite(sensitivity).all())
            self.assertTrue(np.all(sensitivity >= 0))
            self.assertEqual(metadata["events"]["selected_rows"], 6)
            self.assertGreater(metadata["events"]["kept_events"], 0)
            self.assertAlmostEqual(
                float(np.mean(sensitivity, dtype=np.float64)),
                metadata["normalization"]["target_average_sensitivity"],
                places=7,
            )
            self.assertEqual(
                metadata["normalization"]["mode"],
                "uniform_full_support_activity_density",
            )
            self.assertAlmostEqual(
                metadata["normalization"]["source_volume_mm3"], 160.0
            )
            self.assertAlmostEqual(
                metadata["normalization"]["final_integral_over_source_volume"],
                metadata["normalization"]["accepted_events_per_photon"],
                places=7,
            )
            self.assertFalse((output_dir / "checkpoint.npz").exists())
            parsed_metadata = json.loads((output_dir / "run_metadata.json").read_text("utf-8"))
            self.assertEqual(parsed_metadata["dimensions"]["detector_count"], 4)

    def test_legacy_integrated_activity_normalization_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            factor_dir, list_path = self._create_fixture(root, density_basis=False)
            config = SensitivityRunConfig(
                factor_dir=factor_dir,
                compton_paths=(list_path,),
                output_dir=root / "legacy",
                source_photons=1000.0,
                physics=ComptonPhysicsConfig(
                    energy_mev=0.511,
                    energy_threshold_sum_mev=0.46,
                    min_event_effective_support=1.0,
                ),
                system_matrix_path=factor_dir / "SysMat_polar",
                detector_path=factor_dir / "Detector.csv",
                coordinate_path=factor_dir / "coor_polar_full.csv",
                rotation_path=factor_dir / "RotMat_full.csv",
                rotate_num=2,
                batch_size=4,
                device="cpu",
                expected_detector_count=4,
            )
            metadata = run_sensitivity_calculation(config)
            self.assertEqual(
                metadata["normalization"]["mode"],
                "legacy_integrated_activity_per_polar_cell",
            )
            self.assertAlmostEqual(
                metadata["normalization"]["final_average"],
                metadata["normalization"]["accepted_events_per_photon"],
                places=7,
            )

    def test_density_source_volume_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            factor_dir, list_path = self._create_fixture(root)
            config = SensitivityRunConfig(
                factor_dir=factor_dir,
                compton_paths=(list_path,),
                output_dir=root / "bad-volume",
                source_photons=1000.0,
                source_volume_mm3=159.0,
                physics=ComptonPhysicsConfig(energy_mev=0.511),
                system_matrix_path=factor_dir / "SysMat_polar",
                detector_path=factor_dir / "Detector.csv",
                coordinate_path=factor_dir / "coor_polar_full.csv",
                rotation_path=factor_dir / "RotMat_full.csv",
                rotate_num=2,
                device="cpu",
                expected_detector_count=4,
            )
            with self.assertRaisesRegex(ValueError, "covering every polar cell"):
                run_sensitivity_calculation(config)

    def test_detector_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            factor_dir, list_path = self._create_fixture(root)
            config = SensitivityRunConfig(
                factor_dir=factor_dir,
                compton_paths=(list_path,),
                output_dir=root / "result",
                source_photons=1000.0,
                physics=ComptonPhysicsConfig(energy_mev=0.511),
                system_matrix_path=factor_dir / "SysMat_polar",
                detector_path=factor_dir / "Detector.csv",
                coordinate_path=factor_dir / "coor_polar_full.csv",
                rotation_path=factor_dir / "RotMat_full.csv",
                rotate_num=2,
                device="cpu",
                expected_detector_count=10496,
            )
            with self.assertRaisesRegex(ValueError, "Detector count is 4"):
                run_sensitivity_calculation(config)

    def test_project_energy_threshold_defaults(self) -> None:
        self.assertEqual(
            ComptonPhysicsConfig(energy_mev=0.218).resolved_energy_threshold_sum_mev,
            0.18,
        )
        self.assertEqual(
            ComptonPhysicsConfig(energy_mev=0.440).resolved_energy_threshold_sum_mev,
            0.40,
        )
        self.assertEqual(
            ComptonPhysicsConfig(energy_mev=0.511).resolved_energy_threshold_sum_mev,
            0.46,
        )

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.npz"
            generator = torch.Generator(device="cpu")
            generator.manual_seed(123)
            torch.randn(4, generator=generator)
            accumulator = torch.arange(8, dtype=torch.float32)
            diagnostics = BatchDiagnostics(input_events=12, kept_events=5)
            _save_checkpoint(
                checkpoint_path,
                "fingerprint",
                accumulator,
                generator,
                diagnostics,
                processed_events=12,
                completed_batches=3,
                file_index=1,
                file_offset=456,
            )
            expected_after_checkpoint = torch.randn(4, generator=generator)

            restored_generator = torch.Generator(device="cpu")
            restored = _load_checkpoint(
                checkpoint_path,
                "fingerprint",
                torch.device("cpu"),
                restored_generator,
            )
            restored_accumulator, restored_diagnostics = restored[0], restored[1]
            self.assertTrue(torch.equal(restored_accumulator, accumulator))
            self.assertEqual(restored_diagnostics.kept_events, 5)
            self.assertEqual(restored[2:], (12, 3, 1, 456))
            self.assertTrue(
                torch.equal(
                    torch.randn(4, generator=restored_generator),
                    expected_after_checkpoint,
                )
            )

    def test_interrupted_run_resumes_to_identical_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            factor_dir, list_path = self._create_fixture(root)
            interrupted_output = root / "interrupted"
            config = SensitivityRunConfig(
                factor_dir=factor_dir,
                compton_paths=(list_path,),
                output_dir=interrupted_output,
                source_photons=1000.0,
                physics=ComptonPhysicsConfig(
                    energy_mev=0.511,
                    energy_threshold_sum_mev=0.46,
                    min_event_effective_support=1.0,
                ),
                system_matrix_path=factor_dir / "SysMat_polar",
                detector_path=factor_dir / "Detector.csv",
                coordinate_path=factor_dir / "coor_polar_full.csv",
                rotation_path=factor_dir / "RotMat_full.csv",
                rotate_num=2,
                batch_size=3,
                device="cpu",
                seed=11,
                expected_detector_count=4,
                checkpoint_every_batches=1,
                progress_every_batches=10,
            )

            real_accumulate = sensitivity_pipeline.accumulate_event_batch
            call_count = 0

            def interrupt_second_batch(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("intentional test interruption")
                return real_accumulate(*args, **kwargs)

            with mock.patch.object(
                sensitivity_pipeline,
                "accumulate_event_batch",
                new=interrupt_second_batch,
            ):
                with self.assertRaisesRegex(RuntimeError, "intentional test interruption"):
                    run_sensitivity_calculation(config)
            self.assertTrue((interrupted_output / "checkpoint.npz").is_file())

            run_sensitivity_calculation(replace(config, resume=True))
            resumed = np.fromfile(interrupted_output / "Sensi_d", dtype=np.float32)

            uninterrupted_output = root / "uninterrupted"
            run_sensitivity_calculation(replace(config, output_dir=uninterrupted_output))
            uninterrupted = np.fromfile(uninterrupted_output / "Sensi_d", dtype=np.float32)
            np.testing.assert_array_equal(resumed, uninterrupted)


if __name__ == "__main__":
    unittest.main()

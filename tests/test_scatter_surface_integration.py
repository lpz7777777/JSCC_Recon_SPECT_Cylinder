import math
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCATTER_SOURCE = (
    REPO_ROOT
    / "Auxiliary_Studies"
    / "GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
    / "ScatterGen_RayTracing_CircularHole"
    / "scatter.cu"
)


def midpoint_rectangle_solid_angle(distance, half_first, half_second, subdivisions):
    first_step = 2.0 * half_first / subdivisions
    second_step = 2.0 * half_second / subdivisions
    cell_area = first_step * second_step
    total = 0.0
    for first_index in range(subdivisions):
        first = -half_first + (first_index + 0.5) * first_step
        for second_index in range(subdivisions):
            second = -half_second + (second_index + 0.5) * second_step
            radius_squared = distance**2 + first**2 + second**2
            total += distance * cell_area / radius_squared**1.5
    return total


def exact_rectangle_solid_angle(distance, half_first, half_second):
    return 4.0 * math.atan(
        half_first
        * half_second
        / (
            distance
            * math.sqrt(
                distance**2 + half_first**2 + half_second**2
            )
        )
    )


class ScatterSurfaceQuadratureTests(unittest.TestCase):
    def test_near_jscc_back_layer_face_quadrature_converges(self):
        # Adjacent 2 x 6 x 2 mm back-layer crystals have centers 2.1 mm
        # apart, so the target's near face is 1.1 mm from the scatter center.
        exact = exact_rectangle_solid_angle(1.1, 3.0, 1.0)
        coarse = midpoint_rectangle_solid_angle(1.1, 3.0, 1.0, 4)
        default = midpoint_rectangle_solid_angle(1.1, 3.0, 1.0, 8)
        fine = midpoint_rectangle_solid_angle(1.1, 3.0, 1.0, 64)

        self.assertLess(abs(default / exact - 1.0), 0.003)
        self.assertLess(abs(fine / exact - 1.0), 0.0001)
        self.assertLess(abs(default - fine), abs(coarse - fine))

    def test_far_single_sample_is_not_used_for_adjacent_crystals(self):
        source = SCATTER_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"SCATTER_NEAR_TARGET_FACE_SUBDIV", 8', source)
        self.assertIn('"SCATTER_NEAR_TARGET_DISTANCE_FACTOR", 2.0f', source)
        self.assertRegex(
            source,
            r"nearTargetDistanceFactor\s*\*\s*maximum_dimension",
        )

    def test_production_launch_uses_surface_kernel(self):
        source = SCATTER_SOURCE.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r"crystalScatterSurfaceSysMatCuda\s*<<<",
        )
        self.assertNotRegex(
            source,
            r"crystalScatterBoundingSphereLegacyCuda\s*<<<",
        )

    def test_surface_kernel_evaluates_energy_per_subcell(self):
        source = SCATTER_SOURCE.read_text(encoding="utf-8")
        start = source.index("__device__ float integrateIntercrystalTargetSurface")
        end = source.index("__device__ int indexFrombitmap_crystal", start)
        body = source[start:end]
        sample_loop = body.index("for (int second_index")
        scatter_energy = body.index("calculateScatterEnergy", sample_loop)
        window_acceptance = body.index("calculategaussianIntegral", scatter_energy)
        target_chord = body.index("rayBoxChordLength", window_acceptance)
        self.assertLess(scatter_energy, window_acceptance)
        self.assertLess(window_acceptance, target_chord)
        self.assertNotIn("Range_Phi", body)

    def test_optional_component_outputs_cover_all_scatter_terms(self):
        source = SCATTER_SOURCE.read_text(encoding="utf-8")
        self.assertIn('getenv("SCATTER_WRITE_COMPONENTS")', source)
        for filename in [
            "C_intercrystal.sysmat",
            "C_highZ_to_crystal.sysmat",
            "C_local_recoil.sysmat",
            "C_local_self_photoelectric.sysmat",
            "C_collimator_to_crystal.sysmat",
            "C_total.sysmat",
        ]:
            with self.subTest(filename=filename):
                self.assertIn(f'"{filename}"', source)


if __name__ == "__main__":
    unittest.main()

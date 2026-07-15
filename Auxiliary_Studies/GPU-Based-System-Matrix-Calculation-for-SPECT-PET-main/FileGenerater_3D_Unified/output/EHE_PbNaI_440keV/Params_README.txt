EHE_PbNaI_440keV
EHE Pb/NaI conventional SPECT, 440 keV source, automatic 440 photopeak window

geometry_type = ConventionalSPECT
collimator_geometry = Siemens Symbia EHE-style triangular-lattice parallel-hole
collimator_material = Pb
collimator_hole_material = Air/Vacuum
detector_material = NaI
source_energy_keV = 440
relative_FWHM_at_source_energy = 0.140096558
use_forced_energy_window = 0
energy_window_lower_keV = 0
energy_window_upper_keV = 0
save_combined_sysmat = 1
enable_detector_recoil_escape = 1
enable_self_scatter_photopeak = 1
detector_crystal_count = 2312
collimator_hole_count = 1250
collimator_thickness_mm = 50.5
hole_diameter_mm = 2.5
septal_thickness_mm = 3.4

shared_JSCC_detector_and_EHE_collimator_front_face_mm = 198.5
cuda_fov_to_local_y_origin_mm = 223.75
collimator_front_face_mm = 198.5
collimator_back_face_mm = 249

This is a standard photopeak-window parameter set. Use PE and Scatter/combined outputs according to the reconstruction model.
The matrix is per emitted photon at source_energy_keV and does not include 225Ac branching ratio.

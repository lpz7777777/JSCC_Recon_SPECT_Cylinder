JSCC_440keV_to_218keVwin
440 keV source, forced 218 keV window; use Scatter_SysMat as cross-talk

source_energy_keV = 440
detector_material = GAGG
detector_density_g_cm3 = 6.6
shield_material = W
shield_density_g_cm3 = 19.35
relative_FWHM_at_source_energy = 0.140096558
use_forced_energy_window = 1
energy_window_lower_keV = 196.30538
energy_window_upper_keV = 239.69462
save_combined_sysmat = 0

enable_detector_recoil_escape = 1
enable_self_scatter_photopeak = 0

This is a cross-talk parameter set. Run ScatterGen with the 440 keV PE matrix and use Scatter_SysMat_*.sysmat as A(218-window <- 440-source).
Do not use SysMat_withScatter for this cross-talk term, because PEGen does not apply the forced energy window.
The matrix is per emitted 440 keV photon and does not include 225Ac branching ratio. When combining with the 218-window direct term, multiply by Y440, or by Y440/Y218 if the 218 direct term is normalized to unit 218 yield.

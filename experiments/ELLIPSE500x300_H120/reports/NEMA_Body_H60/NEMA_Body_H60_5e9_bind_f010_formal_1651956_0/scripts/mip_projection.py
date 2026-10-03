"""Display-only axial MIP selection; never edit reconstruction arrays."""
import numpy as np

DEFAULT_TRIM_LAYERS = 3


def axial_selection(z_mm, trim_layers=DEFAULT_TRIM_LAYERS):
    z = np.asarray(z_mm)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("MIP needs finite axial centers")
    if not isinstance(trim_layers, (int, np.integer)) or trim_layers < 0 or 2*trim_layers >= len(z):
        raise ValueError("MIP trim must retain at least one axial layer")
    dz = np.diff(z)
    if dz[0] <= 0 or not np.allclose(dz, dz[0]):
        raise ValueError("MIP requires ascending uniform z centers")
    stop = len(z)-trim_layers
    return slice(trim_layers, stop), dict(
        trim_layers_each_end=int(trim_layers), removed_mm_each_end=float(trim_layers*dz[0]),
        retained_indices_zero_based=[int(trim_layers), int(stop-1)],
        retained_layer_count=int(stop-trim_layers),
        retained_center_range_mm=[float(z[trim_layers]), float(z[stop-1])],
        retained_slab_bounds_mm=[float(z[trim_layers]-dz[0]/2),float(z[stop-1]+dz[0]/2)],
        only_mip_display_selection=True, reconstruction_and_quantitative_metrics_unchanged=True)


def axial_mip(volume, z_mm, trim_layers=DEFAULT_TRIM_LAYERS):
    selection, _ = axial_selection(z_mm, trim_layers)
    if np.ndim(volume) != 3 or volume.shape[0] != len(z_mm):
        raise ValueError("MIP expects [z,y,x] matching axial centers")
    return np.max(volume[selection], axis=0)

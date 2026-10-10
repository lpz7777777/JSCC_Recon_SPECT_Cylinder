"""NEMA H60 sphere center margin and one complete-voxel non-insert background."""
import numpy as np

POLICY_ID = 'nema_h60_sphere_center_margin15_common_background_20261010'


def build_masks(truth, manifest, config):
    # The registered inner cross-section is convex, so four interior corners
    # imply a completely interior XY square. Strict tests exclude contact.
    x, y, z = [np.asarray(truth[k + '_mm'], dtype=np.float64) for k in 'xyz']
    pitch = float(config['truth_spacing_mm'])
    if pitch != 3 or any(not np.allclose(np.diff(v), pitch, rtol=0, atol=1e-12) for v in (x, y, z)):
        raise ValueError('Expected the original complete 3 mm truth lattice')
    half = pitch / 2
    xx, yy, zz = x[None, None, :], y[None, :, None], z[:, None, None]
    shape = (len(z), len(y), len(x))
    a = config['body_arc_geometry_mm']
    top, bottom, offset, cy = [float(a[k]) for k in
        ('top_inner_radius', 'bottom_inner_radius', 'bottom_arc_center_offset_x', 'circle_center_y')]
    if not np.isclose(top - bottom, offset):
        raise ValueError('Registered circular arcs are not tangent')
    def inside(px, py):
        dy = py - cy
        upper = (dy >= 0) & (px * px + dy * dy < top * top)
        lower = (dy < 0) & (dy > -bottom) & (np.abs(px) < offset + np.sqrt(np.maximum(bottom * bottom - dy * dy, 0)))
        return upper | lower
    body = np.ones(shape, dtype=bool)
    for dx in (-half, half):
        for dy in (-half, half):
            body &= inside(xx + dx, yy + dy)
    body &= np.abs(zz) + half < float(config['body_height_mm']) / 2
    background = body.copy()
    # Closest cube point (not just its corners): a sphere could intersect a
    # face or even lie inside a voxel despite every corner being outside it.
    lung_radius = float(config['lung_outer_diameter_mm']) / 2
    background &= np.maximum(np.abs(xx) - half, 0)**2 + np.maximum(np.abs(yy) - half, 0)**2 > lung_radius**2
    spheres = {}
    for item in manifest['spheres']:
        d = int(item['diameter_mm']); radius = float(item['diameter_mm']) / 2
        cx, sy, cz = item['center_mm']
        center_distance2 = (xx - cx)**2 + (yy - sy)**2 + (zz - cz)**2
        closest2 = np.maximum(np.abs(xx - cx) - half, 0)**2 + np.maximum(np.abs(yy - sy) - half, 0)**2 + np.maximum(np.abs(zz - cz) - half, 0)**2
        mask = (center_distance2 <= (radius - half)**2) & body
        spheres[d] = mask
        background &= closest2 > radius**2
        if np.any(truth[f'sphere_{d}_fraction_zyx'][mask] <= 0):
            raise ValueError('Interior sphere centers disagree with original source truth')
    if int(background.sum()) < 2:
        raise ValueError('Common background is empty')
    for energy in (218, 440):
        expected = float(config['relative_activity_concentration'][f'background_{energy}'])
        if np.any(truth[f'activity_{energy}_zyx'][background] != expected):
            raise ValueError('Common background includes a non-background truth voxel')
    if any(np.any(background & mask) for mask in spheres.values()):
        raise ValueError('Sphere and common background overlap')
    return dict(background=background, body=body, spheres=spheres, pitch_mm=pitch,
        policy_id=POLICY_ID, sphere_center_margin_mm=half, extra_erosion_mm=0,
        sphere_cube_corners_may_cross_surface=True, background_boundary_contact_excluded=True)


def measure(image, masks, manifest, energy):
    image = np.asarray(image)
    background = image[masks['background']].astype(np.float64)
    if not np.isfinite(background).all() or np.any(background < 0):
        raise ValueError('Invalid common-background values')
    mean = float(background.mean()); sd = float(background.std(ddof=1))
    rows = []
    for item in manifest['spheres']:
        d = int(item['diameter_mm']); roi = masks['spheres'][d]
        values = image[roi].astype(np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid sphere values')
        hot = int(item['hot_energy_keV']) == int(energy)
        h = float(values.mean()) if len(values) else None
        crc = (h / mean - 1) / (9 if hot else -1) if h is not None and mean > 0 else None
        cnr = (h - mean) / sd if h is not None and sd > 0 else None
        rows.append(dict(diameter_mm=d,energy_keV=int(energy),region_kind='hot' if hot else 'cold',
            truth_sphere_background_ratio=10 if hot else 0,sphere_voxels=int(roi.sum()),
            background_voxels=int(background.size),sphere_mean=h,background_mean=mean,background_std=sd,
            crc=crc,cnr=cnr,roi_status='AVAILABLE' if len(values) else 'EMPTY_SPHERE_ROI',
            cnr_status='EMPTY_ROI' if not len(values) else 'ZERO_BACKGROUND_STD' if sd == 0 else 'AVAILABLE'))
    return dict(background_mean=mean,background_std=sd,background_cv=sd/mean if mean > 0 else None,
        background_voxels=int(background.size)), rows

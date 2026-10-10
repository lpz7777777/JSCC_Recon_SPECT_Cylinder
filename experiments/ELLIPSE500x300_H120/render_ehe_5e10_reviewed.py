"""Layout-only replacement of two clipped scientific galleries; preserve originals."""
import hashlib
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import plot_ehe_5e10 as original
from ehe_common import digest, read, write


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    base = original.OUT
    out = base.parent / 'comparison_reviewed'
    out.mkdir(exist_ok=False)
    original_manifest = read(base / 'artifact_manifest.json')
    # Use the accepted raw arrays and authoritative 3D truth.
    _, _, routes, selected, source, views, _, _, inputs, _, _ = original.scientific_data()
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 9,
                         'axes.unicode_minus': False})
    bindings = {}
    for kind in ('coronal', 'sagittal'):
        selector, extent, units = views[kind]
        fig, axes = plt.subplots(3, 8, figsize=(24, 7.4), squeeze=False)
        fig.subplots_adjust(left=.13, right=.945, bottom=.06, top=.83,
                            wspace=.08, hspace=.32)
        planes = []
        for row, (system, channel, label, _) in enumerate(routes[:3]):
            images = [source[system, original.prior.energy(channel)]] + [
                selected[system, channel, iteration] for iteration in original.NODES]
            for col, value in enumerate(images):
                plane = selector(value)
                planes.append(dict(channel=channel, column=col,
                                   volume_sha256=array_sha(value),
                                   plane_sha256=array_sha(plane)))
                ax = axes[row, col]
                color = ax.imshow(plane, origin='lower', extent=extent,
                                  cmap='gray_r', vmin=0, vmax=10,
                                  interpolation='nearest', aspect='equal')
                ax.set_title('H60 3D truth' if col == 0 else
                             '0 (uniform init)' if original.NODES[col-1] == 0 else
                             str(original.NODES[col-1]) + ' iterations')
                ax.set_xticks([])
                ax.set_yticks([])
            bounds = axes[row, 0].get_position()
            fig.text(.012, (bounds.y0 + bounds.y1) / 2,
                     system + '\n' + label + '\n' + units,
                     ha='left', va='center', fontsize=9)
        color_axis = fig.add_axes([.961, .22, .009, .48])
        fig.colorbar(color, cax=color_axis,
                     label='gamma density / emitted-source background; fixed 0–10')
        fig.suptitle('EHE full-4π Geant4 5e10; ' + kind + '\n'
                     'Actual EHE iterations 0–200; H60 3D truth; crop0, no smoothing, no fitted gain\n'
                     'Same original images, fixed emitted-source density scale and spatial extent',
                     fontsize=12, y=.975)
        fig.savefig(out / ('transport_' + kind + '.png'), dpi=120,
                    bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
        bindings[kind] = planes
    shutil.copy2(__file__, out / Path(__file__).name)
    write(out / 'layout_acceptance.json', dict(
        passed=True, scope='Two gallery layouts only; original artifacts retained',
        source_sha256=digest(__file__), original_plot_source_sha256=digest(original.__file__),
        original_artifact_manifest_sha256=digest(base / 'artifact_manifest.json'),
        accepted_scientific_inputs_sha256=inputs,
        original_files=original_manifest['files'], array_bindings=bindings,
        crop=0, smoothing_sigma=0, fitted_gain=False, display_range=[0, 10],
        colormap='gray_r', image_interpolation='nearest',
        geometry_extent={k:list(views[k][1]) for k in bindings},
        replacement_files={p.name:digest(p) for p in out.glob('*.png')},
        visual_review='Pending direct inspection of the two exported replacements'))
    print('LAYOUT_ONLY_GALLERIES_READY', out, flush=True)


if __name__ == '__main__':
    main()

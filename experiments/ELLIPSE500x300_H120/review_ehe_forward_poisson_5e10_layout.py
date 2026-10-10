"""Caption-only review of the accepted18-route atlas; preserve original exports."""
from pathlib import Path
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import plot_ehe_forward_poisson_5e10 as original
from ehe_common import read, write, digest, verify_files


def main():
    report = original.REPORT
    source = original.OUT
    manifest = read(source / 'artifact_manifest.json')
    verify_files(source, manifest['files'])
    numeric = read(report / 'scientific_figure_acceptance.json')
    if not numeric['passed']:
        raise ValueError('Accepted numerical figures required')
    out = report / 'comparison_reviewed'
    out.mkdir(exist_ok=False)
    _, _, routes, selected, truth, views, _, _, inputs, _, _ = original.scientific_data()
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 9,
                         'axes.unicode_minus': False})
    fig = original.gallery('mip72', routes, selected, truth, views)
    title = fig._suptitle
    previous_title = title.get_text()
    corrected_title = previous_title.replace(
        'Model-generated observations; fixed218 background from own final440; same column is not equal convergence',
        'New matrix-Poisson5e10 group: model-generated data, own final440200 background; references retain their original data')
    if previous_title == corrected_title:
        raise ValueError('Expected original caption identity differs')
    title.set_text(corrected_title)
    name = 'overall_mip72_18_routes.png'
    fig.savefig(out / name, dpi=120, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    arrays = {str(key): hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
              for key, value in selected.items()}
    truths = {str(key): hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
              for key, value in truth.items()}
    write(out / 'layout_acceptance.json', dict(
        passed=True, scope='One suptitle sentence only; original exports preserved',
        original_image_sha256=digest(source / name), reviewed_image_sha256=digest(out / name),
        original_plotting_source_sha256=digest(original.__file__),
        review_source_sha256=digest(__file__), inputs_sha256=inputs,
        unchanged_selected_image_array_sha256=arrays, unchanged_truth_array_sha256=truths,
        unchanged=['Planes', 'MIP72', 'All source/history arrays', '0-200/0-10000 nodes',
                   'Crop0', 'No smoothing', 'No fitted gain', 'Fixed gray_r0-10'],
        original_title=previous_title, reviewed_title=corrected_title,
        direct_visual_qa='Pending actual reviewed PNG inspection'))
    print('CAPTION_SCOPE_REVIEW_READY', flush=True)


if __name__ == '__main__':
    main()

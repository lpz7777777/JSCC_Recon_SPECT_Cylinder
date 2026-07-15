# ExportSysMatForAmide

Exports any completed matrix under `runs/<case>/` using the same raw Cartesian
layout as `Factors/*/SysMat_tmp`.

The exporter reads dimensions from `Params_Image.dat`, reads detector flags
from `Params_Detector.dat`, validates the complete matrix byte count, retains
`flag == 1` scintillator rows, and writes the result in streaming blocks.

## Interactive use

From the repository root:

```bash
./export_sysmat_for_amide.sh
```

Select a run and then one of its `.sysmat` files by number. Output defaults to:

```text
AmideExports/<run-name>/<matrix-name>/
```

## MATLAB API

```matlab
addpath('ExportSysMatForAmide');

result = export_run_sysmat_for_amide( ...
    'EHE_PbNaI_440keV', ...
    'SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat');
```

Optional arguments:

```matlab
'FilterScintillator', true   % keep Params_Detector flag==1, like GenFactors
'Overwrite', false           % protect an existing export
'ChunkDetectors', 64         % streaming block size
```

## Output

- `SysMat_tmp`: float32 little-endian raw matrix.
- `AMIDE_IMPORT.txt`: exact AMIDE raw import settings.
- `DetectorIndex.csv`: output-frame to original-detector mapping.
- `metadata.json` and `metadata.mat`: machine-readable dimensions and statistics.

The file dimension order is:

```text
[image_x, image_y, image_z, selected_detector, rotation]
```

For AMIDE raw import, combine detector and rotation into the time/frame axis.
Use the `[X Y Z T]`, voxel size, data type, byte order, and offset values written
to `AMIDE_IMPORT.txt`.

The exporter rejects missing or size-incomplete matrices. Nonfinite input values
are replaced with zero, matching the defensive behavior in `GenFactors`.
# AMIDE System-Matrix Export

`run_export_for_amide` provides an interactive selection of a `runs/<case>`
directory and a `.sysmat` file, then writes an AMIDE-compatible `SysMat_tmp`
with detector filtering and import metadata.

The program accepts a run name, a relative run directory, or an absolute path.
On Windows, drive-qualified paths such as `F:\\data\\runs\\EHE_PbNaI_218keV`
are preserved rather than appended to the current directory. This applies to
the run, input matrix, and output paths.

To run the regression test in MATLAB:

```matlab
cd ExportSysMatForAmide
results = runtests('tests')
```

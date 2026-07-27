# Tools

This directory contains supporting scripts that are not part of the stable reconstruction import path.

## Layout

- `runners/active/`: monitored or repeatable experiment launchers. The current half-split closure runner remains at the repository root until its active job finishes; move it here only after the run is complete.
- `runners/historical/`: launch and monitoring scripts for completed validation campaigns. These scripts resolve the repository root from their installed location.
- `visualization/`: standalone reconstruction, Compton, and sensitivity plotting scripts.
- `diagnostics/`: one-off List and system-response inspection scripts.
- `evaluation/`: MATLAB CRC/CNR, PVR, SSIM, and image-display utilities.
- `legacy/`: superseded reconstruction copies retained for reproducibility.

The stable Python reconstruction modules remain in the repository root because current entry points, distributed jobs, tests, and reproduction scripts import them directly. Moving those modules requires a package migration and coordinated import updates.

Run scripts from the repository root unless their own documentation says otherwise. Historical PowerShell launchers accept `-RepositoryRoot` when an explicit root is needed.

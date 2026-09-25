# Distributed Dual-Energy Compton Reconstruction

This directory freezes the current shared `K*B` Compton response and distributes the same six-output local validation chain across multiple nodes and GPUs.

Outputs:

1. `Image_440_SinglePhoton`
2. `Image_440_ComptonOnly`
3. `Image_440_SinglePlusCompton`
4. `Image_218_SinglePhoton_CrossTalkCorrected`
5. `Image_440SinglePlus218Single`
6. `Image_440SingleComptonPlus218Single`

The 440-to-218 response is a fixed additive Poisson background for the 218
reconstruction. The installed FOV120 `Factors/440keV_RotateNum20/Sensi_d` is
used, with its own grid/physics/hash provenance. In 440 JSCC MLEM,
`Image_440_SinglePlusCompton` updates **one image** using the sum of single-
photon and Compton backprojections and `Sensi_s + Sensi_d`; it is not a sum
of independently reconstructed 440 images. Only the two `Plus218Single`
outputs add the corrected 218 image after reconstruction. Their result is
a gamma-channel composite, not direct parent-225Ac activity.

Geant4 has already applied one Gaussian energy draw per crystal; the List
is passed with `input_energies_already_smeared=True`, avoiding a second random
draw. The 13% relative FWHM at 511 keV (19.903% at 218, 14.010% at 440)
continues to set the Compton cone energy-to-angle uncertainty. Event screening
uses `E1+E2 > 350 keV`. CntStat 218/440 windows are ±half the corresponding
relative FWHM around the photopeak; single-photon Factors analytically
integrate the same Gaussian window acceptance.

Parallel decomposition:

- Direct and cross-talk system matrices are sharded by detector bin.
- Each List CSV is partitioned by byte range on complete line boundaries; ranks do not load and discard the full file.
- The full 440 matrix is replicated once per GPU only for local `K*B` Compton response materialization.
- MLEM backprojections are summed with NCCL `all_reduce`; every rank advances an identical image.
- Rank 0 writes images, iteration histories, predicted cross-talk CntStat, and `run_manifest.json`.

## Current FOV120 resource evidence (2026-09-25)

This loader reads the three complete Factors into host memory on each rank,
then copies the rank's detector shards and a complete 440 matrix to its GPU
for `K*B` Compton event materialization. FOV120 is 51240 voxels (40 x 1281),
10496 crystals and 20 views. Each polar float32 matrix is 2.00 GiB on disk;
the three matrices alone require about 6.01 GiB per rank in host storage,
plus framework and copy overhead. The 60-mm figures based on 25620 points
must not be used for FOV120 memory planning.

Real Contrast 1e9 short job 1626513 ran on four nodes with two RTX5090 GPUs
per node: 194835 accepted Compton events, peak reserved memory ~9.97 GiB/GPU,
6 finite nonnegative finals and valid histories. Full 10000-step Contrast
1626525 and Uniform 1626560 completed in 10m and 13m26s, respectively,
on the same eight-GPU layout. Uniform accepted 194261 events. The respective
440 CntStat counts were 2040723 and 2041123; Compton is ~9.5% of 440 singles
by observation count, not by information weight. Results and limitations are
in [FOV120 status](../../docs/FOV120_EXPERIMENT_STATUS.md).

For 1e10, accepted events could rise by about tenfold, and per-rank event
rows will dominate GPU memory. First measure the actual collected List and
run `preflight.py` with the correct count level, world size and GPU GiB;
keep at least 20% peak memory headroom. Use the 4-node x 2-GPU topology
for 1e9 when nodes have fragmented free GPUs. Increase total ranks or change
topology for 1e10 based on measurements; do not silently reduce the
51240-point grid or sampling strides.

Validation commands from the repository root:

```bash
python distributed/dual_energy_compton_python/preflight.py \
  --experiment-config experiments/FOV120/config.json \
  --factors-dir experiments/FOV120/generated/Factors \
  --cntstat-dir experiments/FOV120/generated/CntStat \
  --list-dir experiments/FOV120/generated/List \
  --data-file-name Contrast --count-level 1e9 \
  --world-size 8 --estimated-accepted-events 194835 --gpu-memory-gib 32
python -m py_compile distributed/dual_energy_compton_python/*.py
torchrun --standalone --nproc_per_node=2 \
  distributed/dual_energy_compton_python/validate_synthetic_distributed.py
```

The synthetic test uses GLOO and compares four distributed MLEM updates against
an unsharded reference, including the fixed 440-to-218 additive background.
See `../dual_energy_compton_slurm/README.md` for cluster commands.

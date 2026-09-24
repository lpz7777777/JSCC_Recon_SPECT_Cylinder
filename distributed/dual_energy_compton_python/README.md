# Distributed Dual-Energy Compton Reconstruction

This directory freezes the current shared `K*B` Compton response and distributes the same six-output local validation chain across multiple nodes and GPUs.

Outputs:

1. `Image_440_SinglePhoton`
2. `Image_440_ComptonOnly`
3. `Image_440_SinglePlusCompton`
4. `Image_218_SinglePhoton_CrossTalkCorrected`
5. `Image_440SinglePlus218Single`
6. `Image_440SingleComptonPlus218Single`

The 440-to-218 response is used as a fixed additive Poisson background. The installed `Factors/440keV_RotateNum20/Sensi_d` is used without recomputation. Energy handling remains 13% FWHM at 511 keV, `E1+E2 >= 350 keV`, and input List energies are treated as already smeared.

Parallel decomposition:

- Direct and cross-talk system matrices are sharded by detector bin.
- Each List CSV is partitioned by byte range on complete line boundaries; ranks do not load and discard the full file.
- The full 440 matrix is replicated once per GPU only for local `K*B` Compton response materialization.
- MLEM backprojections are summed with NCCL `all_reduce`; every rank advances an identical image.
- Rank 0 writes images, iteration histories, predicted cross-talk CntStat, and `run_manifest.json`.

The current implementation intentionally favors a verifiable production path:
each rank reads the three complete Factors into host memory, then copies only
its detector shards plus the complete 440 matrix needed by local Compton events
to its GPU. For 4 nodes x 8 ranks this is roughly 3.2 GiB of matrix host memory
per rank before framework overhead. Run the two-GPU smoke job first and check
both node RAM and GPU memory before increasing the List count level.

Observed full-grid 1e9 runs accept about 185,000 Compton events. If acceptance
scales linearly, 1e10 on 32 GPUs stores about 58,000 normalized event rows per
GPU, or roughly 5.9 GiB at 25,620 float32 pixels, plus the 1.1 GiB full 440
matrix and temporary tensors. The supplied 1e10/32-GPU layout is therefore a
reasonable first production target for 32 GiB GPUs. A 1e11 run with the same
32 ranks would require roughly 59 GiB per GPU for event rows alone and should
not be submitted without more ranks or a streamed/out-of-core response mode.

Validation commands from the repository root:

```bash
python distributed/dual_energy_compton_python/preflight.py --count-level 1e10 --world-size 32
python -m py_compile distributed/dual_energy_compton_python/*.py
torchrun --standalone --nproc_per_node=2 \
  distributed/dual_energy_compton_python/validate_synthetic_distributed.py
```

The synthetic test uses GLOO and compares four distributed MLEM updates against
an unsharded reference, including the fixed 440-to-218 additive background.
See `../dual_energy_compton_slurm/README.md` for cluster commands.

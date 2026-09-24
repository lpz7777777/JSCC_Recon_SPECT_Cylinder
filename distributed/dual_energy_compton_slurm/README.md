# SLURM launchers

The scripts assume the same environment as the existing `distributed/scripts` jobs: CUDA 12.9, Miniforge `torch`, NCCL over `mlx5_bond_0`/`bond0`, and the `gpu_5090` partition.

From the repository root:

```bash
export JSCC_REPO_ROOT="$PWD"
mkdir -p distributed/dual_energy_compton_slurm/logs
python distributed/dual_energy_compton_python/preflight.py --count-level 1e10 --world-size 32
sbatch distributed/dual_energy_compton_slurm/smoke_1node_2gpu.sh
```

After the smoke job writes a valid manifest and six images:

```bash
sbatch distributed/dual_energy_compton_slurm/run_1e10_4nodes_8gpu.sh
```

For a single-node eight-GPU production run, use the standalone launcher:

```bash
python distributed/dual_energy_compton_python/preflight.py --count-level 1e10 --world-size 8
sbatch distributed/dual_energy_compton_slurm/run_1e10_1node_8gpu.sh
```

This launcher uses `torchrun --standalone --nproc_per_node=8`. It does not
require an inter-node rendezvous or `MASTER_NODE`; all other Factors, List,
CntStat, iteration, and frozen-response settings are the same as the 4-node
launcher. Use a separate `--output-dir` when comparing both configurations.

Monitor either a smoke or formal job:

```bash
bash distributed/dual_energy_compton_slurm/monitor.sh JOB_ID
```

The formal template uses 4 nodes x 8 GPUs, full polar Compton sampling, 1000 MLEM iterations, and the installed Half-A-validated `Sensi_d`. Command-line arguments appended to `sbatch ... -- <args>` override or extend the entry-point arguments; for a new output directory pass `--output-dir` explicitly.

This template targets 1e10. Do not change only `--count-level` to 1e11 on the
same 32 GPUs: linear scaling from the measured 1e9 acceptance predicts about
59 GiB of materialized Compton response per GPU before matrices and workspace.

The smoke job still reconstructs the complete 1e10 CntStat arrays, but limits
Compton preprocessing to 256 input rows per view per rank and runs only two
iterations. It validates imports, Factors, detector sharding, NCCL collectives,
cross-talk correction, List parsing, and all six output names. It is not an
image-quality result.

After either job completes, use the existing visualization without changing
the physical response:

```bash
python tools/visualization/compton/visualize_jscc_compton_validation.py \
  --result-dir Results/Reconstruction/Distributed_JSCC_ComptonValidation_Geant4_1e10_Iter1000
```

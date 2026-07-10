"""Multi-energy multi-output task definitions.

This module is pure logic (no torch, no distributed) so it can be unit-tested in
isolation. It turns the five user-facing output types into a flat list of
``ReconTask`` objects that the reconstruction driver consumes.

Output types
------------
1. per-energy single-photon only          -> one "S" task per energy
2. per-energy Compton only                -> one "D" task per whitelisted energy
3. all-energies single-photon only        -> one "S" task over all energies
4. all-energies Compton only              -> one "D" task over all whitelisted energies
5. all-energies joint (S + D)             -> one "J" task (S over all + D over whitelist)

A "whitelisted energy" is one listed in ``--compton-energies``. Energies not in
the whitelist never have their Compton (List) data loaded, so types 2/4/5 simply
do not include them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReconTask:
    """A single reconstruction job.

    Attributes
    ----------
    mode : str
        "S" (single-photon only), "D" (Compton only), or "J" (joint).
    energy_indices : tuple[int, ...]
        Indices (into the full ``e0_list``) of energies whose single-photon
        projection data participates in this task.
    compton_energy_indices : tuple[int, ...]
        Indices of energies whose Compton data participates. Empty for "S"
        tasks; the whole whitelist for "D"/"J" tasks.
    iter_num : int
        Total OSEM iterations for this task.
    save_iter_step : int
        Snapshot interval (must divide ``iter_num``).
    output_name : str
        File-name stem for the saved image (``Image_<output_name>``).
    type_tag : str
        Human-readable source type, e.g. "Type1"/"Type3"/"Type5", for logging.
    """

    mode: str
    energy_indices: tuple
    compton_energy_indices: tuple
    iter_num: int
    save_iter_step: int
    output_name: str
    type_tag: str

    @property
    def dedup_key(self):
        """Identity key used to drop redundant tasks across types.

        Two tasks are redundant iff they have the same mode and the same energy
        participation (both single-photon and Compton subsets). For example, if
        the whitelist has a single energy, Type4 (all-energies Compton) collides
        with Type2 (per-energy Compton) for that energy, so one is dropped.
        """
        return (
            self.mode,
            tuple(sorted(self.energy_indices)),
            tuple(sorted(self.compton_energy_indices)),
        )


def _energy_tag(e0_list, indices):
    """Render energy indices as a keV string, e.g. (440_218)keV or 440keV."""
    kev_list = [str(round(e0_list[i] * 1000)) for i in sorted(indices)]
    if len(kev_list) == 1:
        return f"{kev_list[0]}keV"
    return "(" + "_".join(kev_list) + ")keV"


def make_output_name(mode, e0_list, energy_indices, compton_energy_indices):
    """Build a readable output-file stem that encodes mode + energies."""
    s_tag = _energy_tag(e0_list, energy_indices)
    if mode == "S":
        return f"S_{s_tag}"
    if mode == "D":
        return f"D_{_energy_tag(e0_list, compton_energy_indices)}"
    # joint: show both the single-photon and Compton participation explicitly
    d_tag = _energy_tag(e0_list, compton_energy_indices)
    return f"J_S{s_tag}_D{d_tag}"


@dataclass
class IterConfig:
    """Per-type iteration configuration. ``iter_num <= 0`` disables that type."""

    type1_single_sc: tuple = (0, 1)        # per-energy single-photon
    type2_single_compton: tuple = (0, 1)   # per-energy Compton
    type3_joint_sc: tuple = (0, 1)         # all-energies single-photon
    type4_joint_compton: tuple = (0, 1)    # all-energies Compton
    type5_joint: tuple = (0, 1)            # all-energies joint


def build_tasks(e0_list, compton_eidx_list, iter_cfg):
    """Build the de-duplicated list of ``ReconTask`` objects.

    Parameters
    ----------
    e0_list : list[float]
        All energies (MeV) passed via ``--e0-list``.
    compton_eidx_list : list[int]
        Indices into ``e0_list`` of energies in the Compton whitelist
        (``--compton-energies``).
    iter_cfg : IterConfig
        Per-type ``(iter_num, save_iter_step)`` tuples.

    Returns
    -------
    list[ReconTask]
        Ordered task list with cross-type duplicates removed.
    """
    ne = len(e0_list)
    all_eidx = tuple(range(ne))
    compton_all = tuple(sorted(compton_eidx_list))

    raw_tasks = []

    # --- Type 1: per-energy single-photon ---
    it1, sv1 = iter_cfg.type1_single_sc
    if it1 > 0:
        for ei in all_eidx:
            raw_tasks.append(
                _make("S", (ei,), (), it1, sv1, e0_list, "Type1"))

    # --- Type 2: per-energy Compton ---
    it2, sv2 = iter_cfg.type2_single_compton
    if it2 > 0 and compton_all:
        for ci in compton_all:
            raw_tasks.append(
                _make("D", (), (ci,), it2, sv2, e0_list, "Type2"))

    # --- Type 3: all-energies single-photon ---
    it3, sv3 = iter_cfg.type3_joint_sc
    if it3 > 0 and ne >= 1:
        raw_tasks.append(
            _make("S", all_eidx, (), it3, sv3, e0_list, "Type3"))

    # --- Type 4: all-energies Compton ---
    it4, sv4 = iter_cfg.type4_joint_compton
    if it4 > 0 and len(compton_all) >= 1:
        raw_tasks.append(
            _make("D", (), compton_all, it4, sv4, e0_list, "Type4"))

    # --- Type 5: all-energies joint ---
    it5, sv5 = iter_cfg.type5_joint
    if it5 > 0 and compton_all:
        raw_tasks.append(
            _make("J", all_eidx, compton_all, it5, sv5, e0_list, "Type5"))

    # Deduplicate while preserving order; keep the first occurrence (lowest type
    # number wins, so Type2 is preferred over the identical Type4).
    seen = set()
    tasks = []
    for t in raw_tasks:
        if t.dedup_key in seen:
            continue
        seen.add(t.dedup_key)
        tasks.append(t)
    return tasks


def _make(mode, energy_indices, compton_indices, it, sv, e0_list, type_tag):
    return ReconTask(
        mode=mode,
        energy_indices=tuple(energy_indices),
        compton_energy_indices=tuple(compton_indices),
        iter_num=int(it),
        save_iter_step=int(sv),
        output_name=make_output_name(mode, e0_list, energy_indices, compton_indices),
        type_tag=type_tag,
    )


def format_task_table(tasks, e0_list):
    """Render the task list as a multi-line string for logging on rank 0."""
    header = f"{'#':>2}  {'Type':<6} {'Mode':<4} {'S-energies':<18} {'D-energies':<18} {'Iter':>6} {'Step':>4}  Output"
    lines = [header, "-" * len(header)]
    for i, t in enumerate(tasks):
        s_kev = _energy_tag(e0_list, t.energy_indices) if t.energy_indices else "-"
        d_kev = _energy_tag(e0_list, t.compton_energy_indices) if t.compton_energy_indices else "-"
        lines.append(
            f"{i + 1:>2}  {t.type_tag:<6} {t.mode:<4} {s_kev:<18} {d_kev:<18} "
            f"{t.iter_num:>6} {t.save_iter_step:>4}  {t.output_name}"
        )
    return "\n".join(lines)

"""Replay bounded synthetic traces against the current List contract.

This is a source-level diagnostic, not Geant4 transport or reconstruction.
No production source, List, response, or sensitivity is modified. Deposits are
unsmeared so the event-definition issue is separated from measurement noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class Step:
    crystal: int | None
    process: str
    pre_gamma_mev: float = 0.0
    post_gamma_mev: float = 0.0
    deposit_mev: float = 0.0
    track_id: int = 1


def legacy_list(steps):
    """Mirror positive-deposit/primary branches; NumPhot is never incremented."""
    first, num_compt, flag = -2, 0, False
    energy = {}
    for step in steps:
        if first == -2 and step.deposit_mev > 0 and step.crystal is None:
            first = -1
        elif step.deposit_mev > 0 and step.crystal is not None:
            energy[step.crystal] = energy.get(step.crystal, 0.0) + step.deposit_mev
            if step.track_id == 1:
                if first == -2:
                    first = step.crystal
                    num_compt += step.process == 'compt'
                elif first == step.crystal:
                    num_compt += step.process == 'compt'
                elif not flag:
                    flag = True
    hits = sorted(c for c, e in energy.items() if e > .001)
    accepted = len(hits) == 2 and first in hits and num_compt == 1 and flag
    e1 = energy.get(first, 0.0)
    e2 = sum(energy[c] for c in hits if c != first)
    edge = 2 * .440**2 / (.511 + 2 * .440) - .001
    energy_candidate = accepted and .050 < e1 < edge and e2 > .050 and e1 + e2 > .350
    return dict(list_accepted=accepted, original_energy_candidate=energy_candidate,
                first_crystal=first, counted_compton=num_compt, counted_photoelectric=0,
                e1_mev=e1, e2_mev=e2)


def first_scatter_contract(steps):
    """Explain first-leg assumptions without imposing one interaction in C2.

    For these fully contained synthetic examples only, aggregate E1 is checked
    against the first gamma's energy transfer. Real MC must separately report
    recoil/fluorescence escape, Doppler effects and measurement uncertainty.
    """
    interactions = [s for s in steps if s.track_id == 1 and s.process in
                    ('compt', 'phot', 'Rayl', 'conv')]
    first = next((s for s in interactions if s.crystal is not None), None)
    reasons = []
    if first is None:
        return dict(compatible=False, reasons=['no primary detector interaction'])
    before = interactions[:interactions.index(first)]
    if before:
        reasons.append('interaction before first detector hit')
    if first.process != 'compt':
        reasons.append('first detector interaction is not Compton')
    later_first = [s for s in interactions[interactions.index(first)+1:]
                   if s.crystal == first.crystal]
    if later_first:
        reasons.append('additional gamma interaction in first crystal')
    energy1 = sum(s.deposit_mev for s in steps if s.crystal == first.crystal)
    transfer = first.pre_gamma_mev - first.post_gamma_mev
    if abs(energy1-transfer) > 1e-9:
        reasons.append('aggregate first-crystal deposit differs from first transfer')
    return dict(compatible=not reasons, reasons=reasons, actual_first_crystal=first.crystal,
                first_transfer_mev=transfer, aggregate_first_deposit_mev=energy1)


def cases():
    c1 = [Step(1, 'compt', .440, .340, .001),
          Step(1, 'electron', deposit_mev=.099, track_id=2)]
    c2_abs = [Step(2, 'phot', .340, 0, .001),
              Step(2, 'electron', deposit_mev=.339, track_id=3)]
    backward_energy = .340 / (1 + 2 * .340 / .511)
    return {
        'clean_first_scatter_full_absorption': c1+c2_abs,
        'second_crystal_multiple_interactions': c1+[
            Step(2, 'compt', .340, .200, .001),
            Step(2, 'electron', deposit_mev=.139, track_id=3),
            Step(2, 'phot', .200, 0, .001),
            Step(2, 'electron', deposit_mev=.199, track_id=4)],
        'return_to_first_crystal_photoelectric': c1+[
            Step(2, 'compt', .340, backward_energy, .001),
            Step(2, 'electron', deposit_mev=.340-backward_energy-.001, track_id=3),
            Step(1, 'phot', backward_energy, 0, .001),
            Step(1, 'electron', deposit_mev=backward_energy-.001, track_id=4)],
        'prior_zero_deposit_rayleigh': [Step(None, 'Rayl', .440, .440, 0)]+c1+c2_abs,
        'zero_local_first_compton_with_secondary_deposit': [
            Step(1, 'compt', .440, .340, 0),
            Step(1, 'electron', deposit_mev=.100, track_id=2)]+c2_abs,
        'hidden_first_compton_then_second_in_same_crystal': [
            Step(1, 'compt', .440, .340, 0),
            Step(1, 'electron', deposit_mev=.100, track_id=2),
            Step(1, 'compt', .340, .300, .001),
            Step(1, 'electron', deposit_mev=.039, track_id=3),
            Step(2, 'phot', .300, 0, .001),
            Step(2, 'electron', deposit_mev=.299, track_id=4)],
    }


def main():
    root = Path(__file__).resolve().parents[2]
    source = root/'Geant4Sim/Geant4Code/src/SteppingAction.cc'
    event = root/'Geant4Sim/Geant4Code/src/EventAction.cc'
    active = re.sub(r'/\*.*?\*/', '', source.read_text(encoding='utf-8'), flags=re.S)
    active = re.sub(r'//[^\n]*', '', active)
    assert 'AddNumPhot()' not in active
    assert 'else if(elocal>0' in active and 'GetTrackID()==1' in active
    examples = cases()
    for steps in examples.values():
        assert abs(sum(s.deposit_mev for s in steps)-.440) < 1e-12
        gamma_energy = .440
        for step in steps:
            if step.track_id == 1:
                assert abs(step.pre_gamma_mev-gamma_energy) < 1e-12
                gamma_energy = step.post_gamma_mev
    rows = {name: {'legacy': legacy_list(steps),
                   'first_scatter_contract': first_scatter_contract(steps),
                   'steps': [asdict(s) for s in steps]} for name, steps in examples.items()}
    for row in rows.values():
        def angle(transfer):
            cosine = 1-.511*transfer/((.440-transfer)*.440)
            return math.degrees(math.acos(cosine)) if -1 <= cosine <= 1 else None
        row['first_scatter_contract']['transfer_angle_degrees'] = angle(
            row['first_scatter_contract']['first_transfer_mev'])
        row['legacy']['aggregate_angle_degrees'] = angle(row['legacy']['e1_mev'])
    for name in ('clean_first_scatter_full_absorption', 'second_crystal_multiple_interactions'):
        assert rows[name]['legacy']['original_energy_candidate']
        assert rows[name]['first_scatter_contract']['compatible']
    for name in ('return_to_first_crystal_photoelectric', 'prior_zero_deposit_rayleigh',
                 'hidden_first_compton_then_second_in_same_crystal'):
        assert rows[name]['legacy']['original_energy_candidate']
        assert not rows[name]['first_scatter_contract']['compatible']
    z = rows['zero_local_first_compton_with_secondary_deposit']
    assert not z['legacy']['list_accepted'] and z['first_scatter_contract']['compatible']
    record = {'purpose': 'Synthetic source-contract diagnostic; not MC frequency or reconstruction',
              'smeared': False, 'geometry_ARM_not_evaluated': True,
              'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (source, event)}, 'cases': rows,
              'limitations': 'Toy traces exercise code paths. No rate, actual event genealogy or post-kernel acceptance is inferred.'}
    output = Path(__file__).resolve().parent/'reports/NEMA_Body_H60/process_list_audit/event_contract_cases.json'
    output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in row.items() if k != 'steps'}
                      for name, row in rows.items()}, indent=2))


if __name__ == '__main__':
    main()

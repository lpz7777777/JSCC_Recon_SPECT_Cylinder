"""Held jobs cannot bypass validation; measured-time limits reject invalid data."""
import tempfile
import unittest
from pathlib import Path
from jscc_5e10_common import TOTAL, digest, write
from jscc_5e10_formal_queue_gate import check_gate, measured_limits


class QueueGateTests(unittest.TestCase):
    def test_measured_limits_and_budget(self):
        a = dict(estimated_max_phase_seconds=1000., measured_prepare_seconds=500., estimated_total_seconds=4000.)
        self.assertEqual(measured_limits(a), dict(phase=1800, prepare=1800, minutes=130, total=7620))
        for key, value in [('estimated_total_seconds', float('nan')),
                           ('measured_prepare_seconds', 0), ('estimated_max_phase_seconds', 90000)]:
            bad = dict(a, **{key: value})
            with self.assertRaises(ValueError):
                measured_limits(bad)

    def test_gate_binds_job_contract_acceptance_and_all_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); contract = root/'contract.json'; write(contract, {'science': 'fixed'})
            evidence = root/'run.json'; write(evidence, {'validation': 10})
            auth = dict(passed=True, contract_sha256=digest(contract), actual_primary_photons=TOTAL,
                        nodes=8, gpus_per_node=4, world_size=32, validation_job=101,
                        strict_acceptance_sha256='accepted', result=str(root),
                        evidence_sha256={'run.json': digest(evidence)},
                        estimated_max_phase_seconds=1000., measured_prepare_seconds=500., estimated_total_seconds=4000.)
            authority = root/'authority.json'; write(authority, auth)
            gate = dict(passed=True, strict_fetch_passed=True, formal_job=103, validation_job=101,
                        contract_sha256=digest(contract), authority_file=str(authority),
                        authority_sha256=digest(authority), strict_acceptance_sha256='accepted', limits=measured_limits(auth))
            path = root/'gate.json'; write(path, gate)
            self.assertEqual(check_gate(path, contract, 103), gate)
            for key, value in [('formal_job', 104), ('strict_fetch_passed', False), ('contract_sha256', 'wrong')]:
                write(path, dict(gate, **{key: value}))
                with self.assertRaises(ValueError): check_gate(path, contract, 103)
            write(path, gate); evidence.write_text('changed')
            with self.assertRaises(ValueError): check_gate(path, contract, 103)
            with self.assertRaises(FileNotFoundError): check_gate(root/'missing', contract, 103)


if __name__ == '__main__': unittest.main()

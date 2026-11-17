"""Small local integration regression for first-use channel directories.

Synthetic bytes, mocked network only. Runs the real transfer orchestrator,
ordered fsync merge, receipts, final hash and atomic rename on temporary files.
Does not run a scientific calculation or contact a supercomputer.
"""
from contextlib import nullcontext, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import chunked_transfer_r5 as ct
from common import write, now
from prepare import REPORT


def digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else 'ABSENT'


def rpc(_client, code, timeout=120):
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(code, '<local-rpc>', 'exec'), {})
    return output.getvalue()


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='cart_transfer_directory_r5_') as work:
        root = Path(work)
        evidence = root/'evidence'
        evidence.mkdir()
        write(evidence/'transfer_parallel_benchmark_20261117.json', {'selected_connections': 4})
        write(evidence/'transfer_benchmark_20261117.json', {'selected_compression': 'zlib1'})
        payload = bytes(range(43))
        source = root/'source.bin'
        source.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()
        chunks = [dict(start=start, bytes=count,
                       sha256=hashlib.sha256(payload[start:start+count]).hexdigest())
                  for start, count in [(0, 17), (17, 17), (34, 9)]]

        def plan(path, size, sha256, offset):
            assert path == str(source) and size == len(payload) and sha256 == expected and offset == 0
            return dict(source=path, bytes=size, sha256=sha256, offset=offset,
                        prefix_sha256=hashlib.sha256(b'').hexdigest(), chunks=chunks)

        def bridge(_client, path, source_bytes, ranges, target, size, sha256, compression, **kwargs):
            assert path == str(source) and source_bytes == len(payload)
            raw = b''.join(payload[start:start+count] for start, count in ranges)
            target = Path(target)
            assert target.parent.exists() and not target.exists()
            assert len(raw) == size and hashlib.sha256(raw).hexdigest() == sha256
            target.write_bytes(raw)
            return dict(bytes=size, sha256=sha256, wire_bytes=len(raw))

        ct.REPORT = evidence
        ct.source_plan = plan
        ct.st.connect = lambda: nullcontext(None)
        ct.st.rpc = rpc
        ct.st.checksum = lambda _client, path, **kwargs: digest(path)
        ct.st.prefix_state = lambda _client, path: dict(
            sha256=digest(path), bytes=Path(path).stat().st_size if Path(path).exists() else 0)
        ct.st.bridge = bridge
        for channel in ('A440', 'A218', 'C440to218'):
            target = root/'missing'/'matrices'/channel/'B_detector_zyx.float32'
            assert not target.parent.exists()
            try:
                (target.parent/'.old_parts').mkdir(exist_ok=True)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Old missing-parent failure was not reproduced')
            result = ct.transfer_one(None, str(source), str(target), expected, len(payload))
            assert result['full_uncompressed_SHA_verified']
            assert target.read_bytes() == payload and not Path(str(target)+'.partial').exists()
            assert not list((target.parent/'.stream_parts_20261117').glob('*.bin'))
            assert len(list((target.parent/'.stream_parts_20261117').glob('*.merged.json'))) == 3
            reused = ct.transfer_one(None, str(source), str(target), expected, len(payload))
            assert reused['reused_verified'] and target.read_bytes() == payload
            try:
                ct.transfer_one(None, str(source), str(target), '0'*64, len(payload))
            except ValueError:
                pass
            else:
                raise AssertionError('Altered successful matrix must be rejected')
            checks.append(dict(channel=channel, missing_parent_reproduced=True,
                first_transfer_complete=True, ordered_merge_and_tail=True,
                full_sha=True, successful_reuse_no_retransfer=True,
                changed_sha_rejected=True))
    result = dict(time=now(), passed=True, synthetic_bytes_only=True,
                  real_directory_merge_hash_rename_code=True,
                  mocked_network=True, no_remote_submission=True, checks=checks)
    write(REPORT/'transfer_directory_checks_r5_20261117.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

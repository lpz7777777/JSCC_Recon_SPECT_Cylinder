"""Read-only closure of executed event partitions; does not evaluate a response."""
import numpy as np
from jscc_5e10_common import digest, read


def verify_parts(result, manifest, collection, world=16, views=20):
    expected = {f'{v:02d}_rank{r:02d}' for v in range(1, views + 1) for r in range(world)}
    if set(manifest['part_receipt_sha256']) != expected:
        raise ValueError('Missing/extra executed rank/view receipt')
    if {p.stem for p in (result / 'parts').glob('*.json')} != expected:
        raise ValueError('Actual partition receipt set differs')
    files, counts = {}, []
    raw_total = original_total = removed_total = 0
    for v in range(1, views + 1):
        final = np.load(result / 'selections' / f'{v}.npy', allow_pickle=False)
        cached = np.load(result / 'selected_rows' / f'{v}.npy', allow_pickle=False)
        if final.dtype != np.dtype('<i8') or final.ndim != 1 or cached.dtype != np.dtype('<f4') or cached.shape != (len(final), 4):
            raise ValueError('Unexpected selected index/row array layout')
        cursor = kept_cursor = 0
        for rank in range(world):
            name = f'{v:02d}_rank{rank:02d}'
            receipt_path = result / 'parts' / (name + '.json')
            sha = digest(receipt_path)
            if sha != manifest['part_receipt_sha256'][name]:
                raise ValueError('Executed receipt SHA differs')
            receipt = read(receipt_path)
            keys = ('rank', 'view', 'raw_rows', 'raw_global_offset', 'original_accepted', 'kept', 'removed')
            if any(type(receipt[k]) is not int or receipt[k] < 0 for k in keys):
                raise ValueError('Invalid partition count/identity type')
            if (receipt['rank'], receipt['view'], receipt['raw_global_offset']) != (rank, v, cursor):
                raise ValueError('Rank identity/raw offsets do not close')
            if receipt['kept'] + receipt['removed'] != receipt['original_accepted'] or receipt['original_accepted'] > receipt['raw_rows']:
                raise ValueError('Raw/accepted/kept/removed counts do not close')
            q_difference = receipt['q_chunk_max_absolute_difference']
            if not np.isfinite(q_difference) or not 0 <= q_difference <= 1e-10:
                raise ValueError('Executed stable-q chunk check differs')
            files['parts/' + name + '.json'] = sha
            for suffix, field in (('.npy', 'selection_sha256'), ('_rows.npy', 'selected_rows_sha256')):
                relative = 'parts/' + name + suffix
                actual = digest(result / relative)
                if actual != receipt[field]:
                    raise ValueError('Partition array SHA differs from executed receipt')
                files[relative] = actual
            indices = np.load(result / 'parts' / (name + '.npy'), allow_pickle=False)
            rows = np.load(result / 'parts' / (name + '_rows.npy'), allow_pickle=False)
            count = receipt['kept']
            if indices.dtype != np.dtype('<i8') or indices.shape != (count,) or rows.dtype != np.dtype('<f4') or rows.shape != (count, 4):
                raise ValueError('Partition array type/shape differs')
            if len(indices) and (indices[0] < cursor or indices[-1] >= cursor + receipt['raw_rows'] or np.any(indices[1:] <= indices[:-1])):
                raise ValueError('Out-of-range/overlapping/unordered partition selection')
            end = kept_cursor + count
            if not np.array_equal(indices, final[kept_cursor:end]) or not np.array_equal(rows, cached[kept_cursor:end]):
                raise ValueError('Final arrays differ from complete executed partitions')
            cursor += receipt['raw_rows']
            kept_cursor = end
            original_total += receipt['original_accepted']
            removed_total += receipt['removed']
        if kept_cursor != len(final) or kept_cursor != manifest['events_per_view'][v - 1] or np.any(final[1:] <= final[:-1]):
            raise ValueError('Complete view count/order differs')
        raw_total += cursor
        counts.append(kept_cursor)
    if (raw_total != collection['raw_list_rows'] or original_total != manifest['original_accepted'] or
            removed_total != manifest['removed'] or sum(counts) != manifest['accepted_events'] or
            sum(counts) + removed_total != original_total):
        raise ValueError('All-view raw/accepted/kept/removed totals do not close')
    return dict(passed=True, world_size=world, views=views, receipts=len(expected), raw_rows=raw_total,
                accepted_events=sum(counts), original_accepted=original_total, removed=removed_total,
                events_per_view=counts, files=files, executed_partition_arrays_equal_final_arrays=True,
                selection_math_not_recomputed=True)

"""Execution-only adaptation: two original logical ranks per physical GPU."""
import ast


def replace_once(source, old, new):
    if source.count(old) != 1: raise ValueError('Unexpected frozen source section: '+old[:80])
    return source.replace(old,new,1)


def adapt(source):
    source=replace_once(source,"        return hashlib.file_digest(f, 'sha256').hexdigest()",
        "        h = hashlib.sha256()\n        for block in iter(lambda: f.read(8 << 20), b''):\n            h.update(block)\n        return h.hexdigest()")
    source=replace_once(source,'    assert world == contract[\'ranks\']',
        "    assert world == contract['ranks'] == 8\n    owned_shards = [rank, rank + 8]\n    logical_world = 16")
    source=replace_once(source,'    def blocks(path):','    def blocks(path, logical_rank):')
    source=replace_once(source,'width = (len(raw) + world - 1) // world','width = (len(raw) + logical_world - 1) // logical_world')
    source=replace_once(source,'min(rank * width, len(raw)), min((rank + 1) * width, len(raw))',
        'min(logical_rank * width, len(raw)), min((logical_rank + 1) * width, len(raw))')
    old="""    for t, ids in blocks(args.release/'calibration/legacy_v01.csv'):
        if len(t): calibration += uniform_sensitivity_sum(t, volume)
        calibration_ids.extend(ids.cpu().tolist())"""
    new="""    for logical_rank in owned_shards:
        logical_calibration = torch.zeros_like(calibration)
        for t, ids in blocks(args.release/'calibration/legacy_v01.csv', logical_rank):
            if len(t): logical_calibration += uniform_sensitivity_sum(t, volume)
            calibration_ids.extend(ids.cpu().tolist())
        calibration += logical_calibration
        del logical_calibration"""
    source=replace_once(source,old,new)
    begin=source.index('    for view in range(20):\n        pieces, ids_list')
    end=source.index('    counts = torch.tensor',begin)
    original=source[begin:end]
    grouped=original.replace('blocks(name):','blocks(name, logical_rank):').replace(
        'dict(view=view+1, original_zero_based_rows=ids_list)',
        'dict(logical_rank=logical_rank, view=view+1, original_zero_based_rows=ids_list)').replace(
        "print('RESPONSE_VIEW', rank, view+1,", "print('RESPONSE_VIEW', rank, logical_rank, view+1,")
    grouped='    for logical_rank in owned_shards:\n'+''.join('    '+line if line.strip() else line for line in grouped.splitlines(keepends=True))
    source=source[:begin]+grouped+source[end:]
    old="""            for view in range(20):
                # The historical storage split has no change to subset order.
                for part in torch.chunk(subsets[view][subset], 2, dim=0):
                    weight += point_weight(part, x)"""
    new="""            for logical_index in range(len(owned_shards)):
                logical_weight = torch.zeros_like(x)
                for view in range(20):
                    # Exact original 16-rank membership and two storage halves.
                    for part in torch.chunk(subsets[logical_index*20+view][subset], 2, dim=0):
                        logical_weight += point_weight(part, x)
                weight += logical_weight"""
    source=replace_once(source,old,new)
    source=replace_once(source,'cache_bytes=bytes_needed, gpu=',
        'cache_bytes=bytes_needed, logical_shards=owned_shards, original_logical_world=16, gpu=')
    ast.parse(source)
    return source


def logical_shards(rank):
    if not 0<=rank<8:raise ValueError('Requires 8 physical ranks')
    return [rank,rank+8]

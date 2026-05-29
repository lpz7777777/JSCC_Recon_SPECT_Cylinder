import os

import torch.distributed as dist


def _read_current_rss_bytes():
    status_path = "/proc/self/status"
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return 0
    return 0


def log_cpu_memory_usage(stage_label):
    stats = {
        "rank": int(dist.get_rank()),
        "pid": int(os.getpid()),
        "rss_gb": _read_current_rss_bytes() / (1024 ** 3),
    }

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, stats)

    if dist.get_rank() == 0:
        gathered = sorted(gathered, key=lambda x: x["rank"])
        rss_values = [item["rss_gb"] for item in gathered]
        max_item = max(gathered, key=lambda x: x["rss_gb"])
        min_item = min(gathered, key=lambda x: x["rss_gb"])
        mean_rss = sum(rss_values) / max(len(rss_values), 1)
        print(
            f"[CPU Memory][{stage_label}] "
            f"mean_rss={mean_rss:.2f}GB "
            f"max_rss={max_item['rss_gb']:.2f}GB(rank={max_item['rank']}, pid={max_item['pid']}) "
            f"min_rss={min_item['rss_gb']:.2f}GB(rank={min_item['rank']}, pid={min_item['pid']})"
        )

"""Summarize raw hot-rod CRC over selected MLEM history frames."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NAME = "EllipseContrast_1e9_1641013"
ITERATIONS = (50, 500, 1000, 3000, 10000)
GROUPS = ("center", "long_plus", "long_minus", "short_plus", "short_minus")


def main():
    result = []
    for iteration in ITERATIONS:
        suffix = "contrast_metrics" if iteration == 10000 else f"contrast_iter{iteration:05d}"
        report = json.loads((HERE / "reports" / f"{NAME}_{suffix}.json").read_text())
        if report["iteration"] != iteration or len(report["rows"]) != 30:
            raise ValueError("Incomplete contrast trajectory")
        medians = {group: float(np.median([row["crc"] for row in report["rows"]
                                            if row["group"] == group])) for group in GROUPS}
        result.append({"iteration": iteration, "group_median_crc": medians,
                       "positive_rods": sum(row["crc"] > 0 for row in report["rows"])})
    frame_root = HERE / "generated/HistorySelected" / NAME
    channel_sums = {}
    for channel in ("218_SinglePhoton_CrossTalkCorrected", "440_SinglePlusCompton"):
        channel_sums[channel] = []
        for iteration in ITERATIONS:
            file = frame_root / f"Image_{channel}_iter{iteration:05d}_full.float32"
            data = np.memmap(file, dtype="<f4", mode="r")
            if len(data) != 132040 or not np.isfinite(data).all():
                raise ValueError("Invalid selected history image")
            channel_sums[channel].append(float(data.sum(dtype=np.float64)))
    summary = {"result": NAME, "method": "Local background CRC; selected verified source-history frames; no smoothing",
               "rows": result, "full_image_sums": channel_sums}
    output = HERE / "reports" / f"{NAME}_contrast_iteration_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for group in GROUPS:
        axes[0].plot(ITERATIONS, [row["group_median_crc"][group] for row in result],
                     marker="o", label=group)
    axes[0].set(xscale="log", xlabel="MLEM iteration", ylabel="median hot-rod CRC")
    axes[0].legend(fontsize=7)
    for channel, sums in channel_sums.items():
        axes[1].plot(ITERATIONS, np.array(sums) / sums[0], marker="o", label=channel)
    axes[1].set(xscale="log", xlabel="MLEM iteration", ylabel="image sum / iteration 50")
    axes[1].legend(fontsize=7)
    fig.suptitle("EllipseContrast 10⁹ primaries: selected iteration frames")
    figure = output.with_suffix(".png")
    fig.savefig(figure, dpi=170)
    print(output)
    print(figure)
    for row in result:
        print(row["iteration"], row["group_median_crc"], row["positive_rods"])


if __name__ == "__main__":
    main()

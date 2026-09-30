"""Show intensity drift hidden by independently scaled gallery panels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
GALLERY = HERE / "reports/iteration_galleries"
ITERATIONS = (50, 500, 1000, 3000, 5000, 10000)
LABELS = {"440_SinglePhoton": "440 single", "440_ComptonOnly": "440 Compton",
          "440_SinglePlusCompton": "440 JSCC",
          "218_SinglePhoton_CrossTalkCorrected": "218 corrected",
          "440SinglePlus218Single": "440 single + 218",
          "440SingleComptonPlus218Single": "440 JSCC + 218"}


def main():
    index = json.loads((GALLERY / "index.json").read_text())
    report = []
    for item in index:
        if item["slice"] != "center":
            continue
        source = json.loads((GALLERY / item["metadata"]).read_text())
        if tuple(source["iterations"]) != ITERATIONS or len(source["channels"]) != 6:
            raise ValueError("Incomplete center gallery metadata")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        rows = {}
        for channel, label in LABELS.items():
            frames = source["channels"][channel]
            if len(frames) != len(ITERATIONS):
                raise ValueError(f"Incomplete channel {channel}")
            sums = np.array([frame["slice_sum"] for frame in frames])
            percentiles = np.array([frame["panel_p99_5"] for frame in frames])
            if sums[0] <= 0 or percentiles[0] <= 0:
                raise ValueError("Nonpositive reference amplitude")
            axes[0].plot(ITERATIONS, sums / sums[0], marker="o", label=label)
            axes[1].plot(ITERATIONS, percentiles / percentiles[0], marker="o", label=label)
            rows[channel] = {"slice_sum_over_50": (sums / sums[0]).tolist(),
                             "p99_5_over_50": (percentiles / percentiles[0]).tolist()}
        axes[0].set(xscale="log", xlabel="MLEM iteration", ylabel="slice sum / iteration 50")
        axes[1].set(xscale="log", xlabel="MLEM iteration", ylabel="panel p99.5 / iteration 50")
        for ax in axes:
            ax.axhline(1, color="black", linestyle="--", linewidth=.7)
            ax.legend(fontsize=7)
        fig.suptitle(f"{item['dataset']}: z={item['z_mm']:+.1f} mm, raw image amplitude")
        output = GALLERY / f"{item['result']}_amplitude.png"
        fig.savefig(output, dpi=170)
        plt.close(fig)
        report.append({"dataset": item["dataset"], "result": item["result"],
                       "z_mm": item["z_mm"], "iterations": ITERATIONS,
                       "channels": rows, "figure": output.name})
        print(output)
    (GALLERY / "amplitude_summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

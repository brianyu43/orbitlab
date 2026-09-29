"""Publication-ready visual summary of the frozen three-seed comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p0 import dump, sha

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "confirmation_v1/evaluation/summary.json"
OUT = HERE / "confirmation_v1/figures"


def main():
    data = json.loads(SOURCE.read_text())
    seeds = (770201, 770202, 770203)
    labels = [str(seed)[-3:] for seed in seeds]
    colors = {"baseline_4k": "#557a95", "candidate_16k": "#d97336"}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
    for ax, split, title in ((axes[0], "seen", "Seen combinations"),
                             (axes[1], "ood", "Excluded combinations")):
        baseline = [100 * data["by_data_seed"][str(seed)]["baseline_4k"]["mean_of_three_initializations"][split]["strict_accepted_and_joint"]
                    for seed in seeds]
        candidate = [100 * data["by_data_seed"][str(seed)]["candidate_16k"]["mean_of_three_initializations"][split]["strict_accepted_and_joint"]
                     for seed in seeds]
        x = np.arange(len(seeds))
        for i in range(len(seeds)):
            ax.plot([x[i]-.11, x[i]+.11], [baseline[i], candidate[i]],
                    color="#b2aaa0", linewidth=1.5, zorder=1)
        ax.scatter(x-.11, baseline, color=colors["baseline_4k"], s=64,
                   label="4k steps" if split == "seen" else None, zorder=2)
        ax.scatter(x+.11, candidate, color=colors["candidate_16k"], s=64,
                   label="16k steps" if split == "seen" else None, zorder=2)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Data seed (last 3 digits)")
        ax.set_ylabel("Strict success (%)")
        ax.set_ylim(0, 50)
        ax.grid(axis="y", alpha=.22)
        ax.set_title(title)
    axes[0].legend(frameon=False, loc="upper left")

    ax = axes[2]
    x = np.arange(len(seeds))
    all_tv = [data["by_data_seed"][str(seed)]["candidate_16k"]["all_generation_diversity"]["aggregate"]["scale"]["tv_mean"]
              for seed in seeds]
    strict_tv = [data["by_data_seed"][str(seed)]["candidate_16k"]["strict_subset_diversity"]["aggregate"]["scale"]["tv"]
                 for seed in seeds]
    ax.bar(x-.17, all_tv, width=.32, color="#8bb7a7", label="All outputs")
    ax.bar(x+.17, strict_tv, width=.32, color="#cc708a", label="Strict passes")
    ax.axhline(.20, color="#444444", linestyle="--", linewidth=1.2, label="Pre-set limit")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Data seed (last 3 digits)")
    ax.set_ylabel("Scale total variation (lower is better)")
    ax.set_ylim(0, .34)
    ax.grid(axis="y", alpha=.22)
    ax.set_title("Candidate scale diversity")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("OrbitLab frozen confirmation: 3 data seeds × 3 initializations", fontsize=13)
    fig.text(.5, -.02, "Success points are means across three initializations. Strict-pass diversity also has cells with fewer than 32 samples.",
             ha="center", fontsize=8, color="#555555")
    OUT.mkdir(parents=True, exist_ok=True)
    png, pdf = OUT / "confirmation_overview.png", OUT / "confirmation_overview.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    dump(OUT / "manifest.json", {"summary_sha256": sha(SOURCE),
                                 "source_sha256": sha(Path(__file__)),
                                 "png_sha256": sha(png), "pdf_sha256": sha(pdf),
                                 "scope": "Three synthetic data seeds, three initializations per seed."})
    print(png, pdf)


if __name__ == "__main__":
    main()

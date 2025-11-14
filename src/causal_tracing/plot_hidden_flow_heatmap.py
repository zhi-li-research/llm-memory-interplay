import copy
import os
from math import ceil

from matplotlib import pyplot as plt


def plot_hidden_flow_heatmap(
    differences,
    labels,
    target_ranges,
    low_score=None,
    kind="hidden",
    savepdf=None,
    title=None,
    xlabel=None,
    show=True,
    font_family="",
    color=None,
    tick_interval=5,
    skip_final_labels_portion=0.1,
    ratio=0.15
):
    labels = copy.deepcopy(labels)
    low_score = low_score if low_score else differences.min()
    answer = "AIE"
    for target_range in target_ranges:
        for i in range(*target_range):
            labels[i] = labels[i] + "*"

    color_map = {
        "hidden": "Purples",
        "mlp": "Greens",
        "attention": "Reds",
    }

    with plt.rc_context(rc={"font.family": font_family}):
        fig, ax = plt.subplots(figsize=(5, len(labels) * ratio), dpi=300)
        h = ax.pcolor(
            differences,
            cmap=color or color_map[kind],
            vmin=low_score,
            edgecolors="#00000010",
        )
        layers = differences.shape[1]
        skip_labels = ceil(layers * skip_final_labels_portion)
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, layers - skip_labels, tick_interval)])
        ax.set_xticklabels(list(range(0, layers - skip_labels, tick_interval)))
        ax.set_yticklabels(labels)
        ax.set_title(f"Impact of restoring {kind} after corrupted input")
        ax.set_xlabel(f"center of interval of restored {kind} layers")
        cb = plt.colorbar(h)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        elif answer:
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        if show:
            plt.show()
        return fig

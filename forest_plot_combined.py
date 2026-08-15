# Makes a color-coded forest plot (AI-generated)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib.ticker as plticker
import warnings

plt.style.use('publication.mplstyle')
warnings.filterwarnings("ignore")

def forest_plot(dirname):
    all_cox = pd.read_csv(
        f"results/{dirname}/statistics/cox_accuracy.csv"
    )

    def ci_position(lower, upper, ref=1.0):
        """
        Return relative position of ref to CI:
        - 'below': ref is below the CI  -> CI entirely above ref
        - 'in':    ref is inside the CI
        - 'above': ref is above the CI  -> CI entirely below ref
        """
        if upper < ref:
            return "above"
        elif lower > ref:
            return "below"
        else:
            return "in"

    # Classify each row for TMLE and HSA
    all_cox["tmle_ci_pos"] = all_cox.apply(
        lambda r: ci_position(r["HR_hsa_tmle_lower"], r["HR_hsa_tmle_upper"]), axis=1
    )
    all_cox["hsa_ci_pos"] = all_cox.apply(
        lambda r: ci_position(r["HR_hsa_js_lower"], r["HR_hsa_js_upper"]), axis=1
    )

    # Combined 9-group label
    all_cox["group"] = all_cox["tmle_ci_pos"] + " / " + all_cox["hsa_ci_pos"]

    # Sort by the 9 groups, then HR_hsa_js (keeps colors clustered together)
    group_order = [
        "below / below", "below / in", "below / above",
        "in / below",    "in / in",    "in / above",
        "above / below", "above / in", "above / above"
    ]
    all_cox["group"] = pd.Categorical(all_cox["group"], categories=group_order, ordered=True)
    all_cox = all_cox.sort_values(["group", "HR_hsa_js"]).reset_index(drop=True)

    # 9 colors, one per group
    group_colors = {
        "below / below": "#1b9e77",
        "below / in":    "#66a61e",
        "below / above": "#a6d854",
        "in / below":    "#7570b3",
        "in / in":       "#bdbdbd",
        "in / above":    "#e6ab02",
        "above / below": "#e7298a",
        "above / in":    "#fc8d62",
        "above / above": "#d95f02",
    }

    row_colors = all_cox["group"].map(group_colors).values
    labels = all_cox["combo"].values
    y_pos = np.arange(len(all_cox))

    def draw_panel(ax, hr_col, colors):
        hr = all_cox[hr_col].values
        lo = all_cox[f"{hr_col}_lower"].values
        hi = all_cox[f"{hr_col}_upper"].values
        for i in range(len(all_cox)):
            ax.errorbar(
                x=hr[i],
                y=y_pos[i],
                xerr=np.array([[hr[i] - lo[i]], [hi[i] - hr[i]]]),
                color=colors[i],
                capsize=3,
                linestyle='None',
                linewidth=1,
                marker="o",
                markersize=4,
                mfc=colors[i],
                mec=colors[i],
            )

    fig, axes = plt.subplots(1, 3, figsize=(9, 8), dpi=300, constrained_layout=True)

    # TMLE and HSA use the group colors; ADD is all black
    draw_panel(axes[0], "HR_hsa_tmle", row_colors)
    axes[0].set_title('HSA TMLE')

    draw_panel(axes[1], "HR_hsa_js", row_colors)
    axes[1].set_title('HSA Joint Sampling')

    draw_panel(axes[2], "HR_add_js", ["black"] * len(all_cox))
    axes[2].set_title('Additivity Joint Sampling')

    # Only the leftmost panel gets y tick labels
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    for ax in axes[1:]:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])

    for ax in axes:
        ax.axvline(x=1, linewidth=0.8, linestyle='--', color='red', alpha=0.5)
        ax.set_xscale('log', base=2)
        x_major = [0.25, 0.5, 1, 2]
        ax.xaxis.set_major_locator(plticker.FixedLocator(x_major))
        ax.xaxis.set_major_formatter(plticker.FixedFormatter(x_major))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
        ax.xaxis.set_minor_formatter(plticker.NullFormatter())
        ax.set_xlim(0.25, 3)

    return fig

dirname = "no_sequencing"
forest_plot(dirname).savefig(
    f"results/{dirname}/statistics/forest_plot_combined.png"
)

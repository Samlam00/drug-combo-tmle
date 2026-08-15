# Creates a plot of predictions vs observed survival

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from lifelines import KaplanMeierFitter

plt.style.use('publication.mplstyle')

def plot_survivals(df_tmle, df_js, df_obs, ax, label=None):
    ticks = [0, 0.5, 1.0]
    TMAX = 73

    # Give labels only once per artist type
    ax.plot(
        df_tmle['Time in months'],
        df_tmle['hsa_tmle_cummin'],
        linewidth=1,
        label='HSA-TMLE Predicted'
    )
   
    kmf_hsa = KaplanMeierFitter()
    kmf_hsa.fit(df_js["hsa_js"])
    kmf_hsa.plot(ax=ax, ci_show=False, legend=False, label='HSA Joint Sampling Predicted', linewidth=1)

    kmf_add = KaplanMeierFitter()
    kmf_add.fit(df_js["add_js"])
    kmf_add.plot(ax=ax, ci_show=False, legend=False, label='Additivity Joint Sampling Predicted', linewidth=1)

    kmf_obs = KaplanMeierFitter()
    kmf_obs.fit(df_obs["OS proxy"], df_obs["os_dx_status"])
    kmf_obs.plot(ax=ax, ci_show=False, legend=False, label='Observed Survival', linewidth=1)

    if label is not None:
        ax.text(-0.1, 1.15, label, transform=ax.transAxes,
                fontweight='bold', va='top', ha='right')
    ax.set_xlabel('')
    ax.set_xlim(0, TMAX - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(6))
    ax.axes.xaxis.set_ticklabels([])

    return ax

def survival_fig():
    # Make everything bigger here
    fig, axes = plt.subplots(
        4, 8,
        figsize=(14, 10),   # <- increase this
        subplot_kw=dict(box_aspect=0.7),
        sharey=True,
        constrained_layout=True
    )
    flat_axes = axes.flatten()

    combos = pd.read_csv("data/metadata/combos.csv")
    crc = pd.read_csv("data/processed_data/crc.csv")
    for i, row in combos.iterrows():
        t1_name = row["t1"]
        t2_name = row["t2"]
        filename = row["filename"]

        df_tmle = pd.read_csv(f"results/{dirname}/{filename}")
        df_js = pd.read_csv("results/joint_sampling/" + filename)
        df_obs = crc[crc["Treatment"] == row["t12"]]

        label = t1_name + " + " + t2_name
        plot_survivals(df_tmle, df_js, df_obs, flat_axes[i], label=None)
        flat_axes[i].text(
            0.85, 0.8, str(i+1),
            transform=flat_axes[i].transAxes,
            fontweight='bold'
        )

    for k in range(17, 20):
        flat_axes[k].axes.xaxis.set_ticklabels([])

    # One legend for the whole figure
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.02)
    )

    return fig

dirname = "tmle"
survival_fig().savefig(
    f"results/{dirname}/statistics/survival_fig.png",
    dpi=300,
    bbox_inches='tight'
)


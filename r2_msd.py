# Compute and plot r^2 and MSD of prediction methods

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from lifelines import KaplanMeierFitter
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

plt.style.use('publication.mplstyle')

def compute_r2_msd(dirname, n=5000):
    combos = pd.read_csv("data/metadata/combos.csv")
    crc = pd.read_csv("data/processed_data/crc.csv")
    scores = [] # per-combo scores

    for i, row in combos.iterrows():
        df_tmle = pd.read_csv("results/" + dirname + "/" + row["filename"])[["Time in months", "hsa_tmle_cummin"]].dropna()
        df_shuffle = pd.read_csv("results/joint_sampling/" + row["filename"])
        df_obs = crc[crc["Treatment"] == row["t12"]]

        f_tmle = interp1d(df_tmle["Time in months"], df_tmle["hsa_tmle_cummin"], kind="nearest", fill_value="extrapolate")

        kmf_hsa = KaplanMeierFitter()
        kmf_hsa.fit(df_shuffle["hsa_js"])
        f_hsa = kmf_hsa.predict

        kmf_add = KaplanMeierFitter()
        kmf_add.fit(df_shuffle["add_js"])
        f_add = kmf_add.predict

        kmf_obs = KaplanMeierFitter()
        kmf_obs.fit(df_obs["OS proxy"], df_obs["os_dx_status"])
        f_obs = kmf_obs.predict

        tgrid = np.linspace(0, row["tmax"]-1, n)

        tmle = f_tmle(tgrid)
        hsa = f_hsa(tgrid)
        add = f_add(tgrid)
        obs = f_obs(tgrid)

        tmle_r2 = r2_score(obs, tmle)
        hsa_r2 = r2_score(obs, hsa)
        add_r2 = r2_score(obs, add)

        tmle_msd = np.sum(obs - tmle) / n
        hsa_msd = np.sum(obs - hsa) / n
        add_msd = np.sum(obs - add) / n

        combo_name =  row["t1"] + " + " + row["t2"]
        scores.append({"combo": combo_name.capitalize(), "hsa_tmle r^2": tmle_r2, "hsa_js r^2": hsa_r2, "add_js r^2": add_r2,
                       "hsa_tmle msd": tmle_msd, "hsa_js msd": hsa_msd, "add_js msd": add_msd})
       
    return pd.DataFrame(scores)

def plot_overall_curve_fit(df):
    fig, axes = plt.subplots(3,2)

    axes[0,0].hist(df["hsa_tmle r^2"], label="HSA-TMLE R^2")
    axes[1,0].hist(df["hsa_js r^2"], label="HSA Joint Sampling R^2")
    axes[2,0].hist(df["add_js r^2"], label="Additivity Joint Sampling R^2")
    axes[0,1].hist(df["hsa_tmle msd"], label="HSA-TMLE MSD")
    axes[1,1].hist(df["hsa_js msd"], label="HSA Joint Sampling MSD")
    axes[2,1].hist(df["add_js msd"], label="Additivity Joint Sampling MSD")
    return fig

r2_msd_statistics = compute_r2_msd("tmle")
r2_msd_statistics.to_csv("results/statistics/r2_msd.csv")

plot_overall_curve_fit(r2_msd_statistics).savefig("results/statistics/r2_msd.png")


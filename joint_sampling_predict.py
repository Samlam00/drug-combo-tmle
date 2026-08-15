# Makes combination survival predictions using HSA-joint-sampling and additivity-joint-sampling methods

import pandas as pd
from scipy.stats import spearmanr
from lifelines import KaplanMeierFitter
import numpy as np
from scipy.interpolate import interp1d
import categorical_encoders as categorical_encoders

SEED = 30000

def fit_rho_2_lists(a, b, rho, rng, ori_rho=None):
    """Shuffles two datasets so that every pair has approximately
    the desired Spearman correlation rho. Copied directly from
    https://github.com/palmerlabunc/clinical-additivity/blob/main/src/utils.py"""

    if ori_rho is None:
        ori_rho = rho
   
    n = len(a)
    pearson_r = 2 * np.sin(rho * np.pi / 6)
    rho_mat = np.array([[1, pearson_r], [pearson_r, 1]])
    size = rho_mat.shape[0]
    means = np.zeros(size)
    u = rng.multivariate_normal(means, rho_mat, size=n)
    i1 = np.argsort(u[:, 0])
    i2 = np.argsort(u[:, 1])
    x1, x2 = np.zeros(n), np.zeros(n)
    x1[i1] = a
    x2[i2] = b

    # check if desired rho is achieved
    result, _ = spearmanr(x1, x2)
    # recursive until reaches 2 decimal point accuracy
    if ori_rho - result > 0.01:  # aim for higher rho
        x1, x2 = fit_rho_2_lists(a, b, rho + 0.01, rng, ori_rho=ori_rho)
    elif ori_rho - result < -0.01:  # aim for lower rho
        x1, x2 = fit_rho_2_lists(a, b, rho - 0.01, rng, ori_rho=ori_rho)

    return (x1, x2)

def fit_rho_3_lists(a, b, c, rho, rng, ori_rho=None):
    """Shuffles three sorted datasets so that every pair has approximately
    the desired Spearman correlation rho. AI-Generated."""
    if ori_rho is None:
        ori_rho = rho

    n = len(a)
    pearson_r = 2 * np.sin(rho * np.pi / 6)
    # 3x3 equicorrelation matrix must stay PSD: off-diagonal >= -1/(k-1) = -0.5
    pearson_r = np.clip(pearson_r, -0.5, 1.0)
    rho_mat = np.array([
        [1.0,       pearson_r, pearson_r],
        [pearson_r, 1.0,       pearson_r],
        [pearson_r, pearson_r, 1.0],
    ])
    size = rho_mat.shape[0]
    means = np.zeros(size)
    u = rng.multivariate_normal(means, rho_mat, size=n)

    i1 = np.argsort(u[:, 0])
    i2 = np.argsort(u[:, 1])
    i3 = np.argsort(u[:, 2])
    x1, x2, x3 = np.zeros(n), np.zeros(n), np.zeros(n)
    x1[i1] = a
    x2[i2] = b
    x3[i3] = c

    # achieved pairwise Spearman correlations
    r12, _ = spearmanr(x1, x2)
    r13, _ = spearmanr(x1, x3)
    r23, _ = spearmanr(x2, x3)
    result = np.mean([r12, r13, r23])

    # recurse until within 2-decimal accuracy, guarding the copula rho range
    if ori_rho - result > 0.01 and rho < 1.0:      # aim for higher rho
        x1, x2, x3 = fit_rho_3_lists(a, b, c, rho + 0.01, rng, ori_rho=ori_rho)
    elif ori_rho - result < -0.01 and rho > -1.0:  # aim for lower rho
        x1, x2, x3 = fit_rho_3_lists(a, b, c, rho - 0.01, rng, ori_rho=ori_rho)

    return (x1, x2, x3)

def sample_times(df, n=5000, times="OS proxy", events="os_dx_status"):
    """Makes a sample of n survival times from the distribution of real patient survival times and events"""
    kmf = KaplanMeierFitter()
    if events == None:
        kmf.fit(durations=df[times])
    else:
        kmf.fit(durations=df[times], event_observed=df[events])

    max_t = df[times].max()

    time = np.arange(0, max_t, 0.1)
    survival = kmf.predict(time).values

    min_surv = survival.min()

    surv_rev = survival[::-1]
    time_rev = time[::-1]

    surv_unique, idx = np.unique(surv_rev, return_index=True)
    time_unique = time_rev[idx]

    f = interp1d(
        surv_unique,
        time_unique,
        kind="nearest",
        bounds_error=False,
        fill_value=max_t,
    )

    percentiles = np.linspace(0, 1, n)

    sampled_times = f(np.maximum(percentiles, min_surv))

    sampled_times[percentiles < min_surv] = max_t

    return sampled_times

def joint_sampling(df1, df2, df12, rho, n=None):
    sample_times1 = sample_times(df1, n)
    sample_times2 = sample_times(df2, n)

    # HSA
    trt1, trt2 = fit_rho_2_lists(sample_times1, sample_times2, rho, np.random.default_rng(SEED))
    hsa = np.sort(np.maximum(trt1, trt2))

    # Additivity

    # sample n untreated times for additivity
    untreated_raw = pd.read_csv("data/raw_data/untreated_raw.csv")
    untreated_f = interp1d(x=untreated_raw["y"], y=untreated_raw["x"], kind="nearest", bounds_error=False, fill_value="extrapolate")
    untreated_times = untreated_f(np.linspace(0, 1, n))

    trt1, trt2, untrt = fit_rho_3_lists(sample_times1, sample_times2, untreated_times, rho, np.random.default_rng(SEED))
    add = np.sort(np.maximum(trt1 + trt2 - untrt, 0))

    # diagnostic for untreated time being larger than treated times
    large_untrt1 = sum(a > b or a > c for a, b, c in zip(untrt, trt1, trt2))
    print(large_untrt1 / len(untrt))
   
    # need this for the cox model censoring
    tmax = np.min([df1["OS proxy"].max(), df2["OS proxy"].max(), df12["OS proxy"].max()])
    hsa = np.clip(hsa, 0, tmax)
    add = np.clip(add, 0, tmax)

    return pd.DataFrame({"hsa_js": hsa, "add_js": add}), tmax

def joint_sampling_predict(dirname, n=None, combo_rho=0.3):
    """Predict survival times for all combos"""
    crc = pd.read_csv("data/processed_data/crc.csv")
    crc["os_dx_status"] = crc["os_dx_status"].apply(categorical_encoders.event_status)
    combos = pd.read_csv("data/metadata/combos.csv")
    combos["tmax"] = 0

    for index, row in combos.iterrows():
        print("predicting for combo " + row["t12"])
        df1 = crc[crc["Treatment"] == row["t1"]]
        df2 = crc[crc["Treatment"] == row["t2"]]
        df12 = crc[crc["Treatment"] == row["t12"]]

        prediction_name = row["t1"] + " + " + row["t2"] + ".csv"
   
        df_both, tmax = joint_sampling(df1, df2, df12, combo_rho, n)
        df_both.to_csv("results/" + dirname + "/" + prediction_name)

        combos.loc[index, "tmax"] = tmax

    # remove extra index columns
    actual_cols = ["t1", "t2", "t12", "A Drug Count", "B Drug Count", "AB Drug Count", "filename", "tmax"]
    combos[actual_cols].to_csv("data/metadata/combos.csv")

joint_sampling_predict("joint_sampling", n=5000, combo_rho=0.3)


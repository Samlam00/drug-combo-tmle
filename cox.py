# Cox proportional hazards test

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from scipy.interpolate import interp1d

SEED = 30000

def tmle_cox_test(dirname, n=500): # the n being used here are how many actual t12 patients there are, so 2n total
    combos = pd.read_csv("data/metadata/combos.csv")
    crc = pd.read_csv("data/processed_data/crc.csv")
    combo_hazard_ratios = []
    for i, row in combos.iterrows():
        df_tmle = pd.read_csv("results/" + dirname + "/" + row["filename"])

        min_survival = min(df_tmle["hsa_tmle_cummin"].dropna())
        f = interp1d(df_tmle["hsa_tmle_cummin"], df_tmle["Time in months"], kind="nearest", bounds_error=False, fill_value=row["tmax"])
        tmle_events = np.hstack((np.repeat(1, int(np.round(n * (1 - min_survival), 0))),
                            np.repeat(0, int(np.round(n * min_survival, 0)))))

        # this is just for off-by-one integer rounding error
        if len(tmle_events) > n:
            tmle_events = tmle_events[len(tmle_events) - n:]
        if len(tmle_events) < n:
            tmle_events = np.insert(tmle_events, 0, np.repeat(0, n - len(tmle_events)))
       
        tmle_times = f(np.linspace(1, 0, n))
        tmle_covariates = list(np.zeros_like(tmle_times))
        tmle = pd.DataFrame({"Times": tmle_times, "Events": tmle_events, "Covariates": tmle_covariates})

        df_obs = crc[crc["Treatment"] == row["t12"]]
        observed_times = df_obs["OS proxy"]
        observed_events = df_obs["os_dx_status"]
        observed_covariates = np.ones_like(observed_times) # in Hwangbo, HR<1 means actual was better than predicted
        observed = pd.DataFrame({"Times": observed_times, "Events": observed_events, "Covariates": observed_covariates})

        cph = CoxPHFitter()
        cph.fit(df=pd.concat([tmle, observed]), duration_col="Times", event_col="Events")
        hr = cph.hazard_ratios_.iloc[0]
        ci_lower = np.exp(cph.confidence_intervals_.iloc[0,0])
        ci_upper = np.exp(cph.confidence_intervals_.iloc[0,1])
        combo_name = row["t1"] + " + " + row["t2"]
        combo_hazard_ratios.append({"combo": combo_name.capitalize(), "HR_hsa_tmle": hr,
                                    "HR_hsa_tmle_lower": ci_lower, "HR_hsa_tmle_upper": ci_upper})
   
    return pd.DataFrame(combo_hazard_ratios)

def js_cox_test(df, df_obs, tmax, model, n=500):
    """Cox proportional hazards test for joint-sampling method predictions"""

    # need to compute min_survival in order to make censoring consistent with Hwangbo et al
    predicted_times = np.sort(df[model])
    survivals = np.linspace(1, 0, len(predicted_times))
    tmp = pd.DataFrame({"Time": predicted_times, "Survival": survivals})
    tmp = tmp[tmp["Time"] < tmax]
    min_survival = tmp["Survival"].min()
    f = interp1d(tmp["Survival"], tmp["Time"], kind="nearest", fill_value="extrapolate")

    predicted_events = np.hstack((np.repeat(1, int(np.round(n * (1 - min_survival), 0))),
                            np.repeat(0, int(np.round(n * min_survival)))))

    # this is just for off-by-one integer rounding error
    if len(predicted_events) > n:
        predicted_events = predicted_events[len(predicted_events) - n:]
    if len(predicted_events) < n:
        predicted_events = np.insert(predicted_events, 0, np.repeat(0, n - len(predicted_events)))
   
    rng = np.random.default_rng(SEED) # add tiny noise for stability
    predicted_times = f(np.linspace(1, 0, n)) + rng.uniform(-1e-6, 1e-6, size=n)
    predicted_covariates = np.zeros_like(predicted_times)
    predicted = pd.DataFrame({"Times": predicted_times, "Events": predicted_events, "Covariates": predicted_covariates})

    observed_times = df_obs["OS proxy"]
    observed_events = df_obs["os_dx_status"]
    observed_covariates = np.ones_like(observed_times)
    observed = pd.DataFrame({"Times": observed_times, "Events": observed_events, "Covariates": observed_covariates})

    cph = CoxPHFitter()
    cph.fit(df=pd.concat([predicted, observed]), duration_col="Times", event_col="Events")
    hr = cph.hazard_ratios_.iloc[0]
    ci_lower = np.exp(cph.confidence_intervals_.iloc[0,0])
    ci_upper = np.exp(cph.confidence_intervals_.iloc[0,1])
    return hr, ci_lower, ci_upper

def hsa_add_js_cox_test(n=500):
    combos = pd.read_csv("data/metadata/combos.csv")
    crc = pd.read_csv("data/processed_data/crc.csv")
    combo_hazard_ratios = []
    for i, row in combos.iterrows():
        df_js = pd.read_csv("results/joint_sampling/" + row["filename"])
        df_obs = crc[crc["Treatment"] == row["t12"]]

        hsa_hr, hsa_ci_lower, hsa_ci_upper = js_cox_test(df_js, df_obs, row["tmax"], "hsa_js", n)
        add_hr, add_ci_lower, add_ci_upper = js_cox_test(df_js, df_obs, row["tmax"], "add_js", n)

        combo_name = row["t1"] + " + " + row["t2"]
        combo_hazard_ratios.append({"combo": combo_name.capitalize(), "HR_hsa_js": hsa_hr,
                                    "HR_hsa_js_lower": hsa_ci_lower, "HR_hsa_js_upper": hsa_ci_upper,
                                    "HR_add_js": add_hr, "HR_add_js_lower": add_ci_lower, "HR_add_js_upper": add_ci_upper})
   
    return pd.DataFrame(combo_hazard_ratios)

dirname = "no_sequencing"
tmle_cox = tmle_cox_test(dirname)
js_cox = hsa_add_js_cox_test()
cox = pd.merge(tmle_cox, js_cox, on="combo")
cox.to_csv("results/" + dirname + "/statistics/cox_accuracy.csv")
print(cox)

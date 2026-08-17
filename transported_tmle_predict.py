# Makes combination survival predictions using HSA-TMLE method

import pandas as pd
import numpy as np
from transported_tmle_truncated import TMLE
import pseudo_values as pseudo_values
import categorical_encoders
import json


def estimate_counterfactuals(t1, t2, obs, t, Wcols):
    """Estimate counterfactual survival for all target at t months for two monotherapies using monotherapy data"""
    print("Estimating counterfactuals for t = " + str(t))

    t1["Pseudo values"] = pseudo_values.compute_pseudo_values(t1, t)
    t2["Pseudo values"] = pseudo_values.compute_pseudo_values(t2, t)

    df = pd.concat([t1, t2]).reset_index(drop=True)

    W = df[Wcols].copy().astype(float)
    A = df["Treatment"]
    Y = df["Pseudo values"]
    W_target = obs[Wcols].copy().astype(float)

    tmle_model = TMLE()
    tmle_model.fit(W, A, Y, W_target)

    # now we need to plug in the new patient covariates
    counterfactual_estimates = tmle_model.predict(obs[Wcols].copy().astype(float))
    counterfactual_estimates = counterfactual_estimates.join(obs) # append patient data for exploratory analysis

    q0 = tmle_model.predict_Q0(W_target)
    rho = q0["t1"].corr(q0["t2"])

    return counterfactual_estimates, rho, tmle_model.r

def predict_survival(crc, t1_name, t2_name, t12_name, Wcols):
    """Predict a survival curve using monotherapy data."""

    t1_crc = crc[crc["Treatment"] == t1_name].reset_index(drop=True).copy()
    t2_crc = crc[crc["Treatment"] == t2_name].reset_index(drop=True).copy()
   
    t1_max = t1_crc["OS proxy"].max()
    t2_max = t2_crc["OS proxy"].max()

    df = categorical_encoders.encode(pd.concat([t1_crc, t2_crc]), t1_name, t2_name)
    # some covariates have multiple dummy columns
    with open("dummy_dict.json", "r") as f:
        dummy_dict = json.load(f)
    expanded_Wcols = []
    for col in Wcols:
        if col in dummy_dict:
            expanded_Wcols.extend([dummy for dummy in dummy_dict[col] if dummy in df.columns])
        elif col in df.columns:
            expanded_Wcols.append(col)

   
    t1 = df[df["Treatment"] == 0].reset_index(drop=True).copy()
    t2 = df[df["Treatment"] == 1].reset_index(drop=True).copy()

    # we need to transport predictions patient population of A+B
    obs = crc[crc["Treatment"] == t12_name].copy() # this is a datatable of the covariates, OS proxy and event status of A+B patients
    obs = categorical_encoders.encode(obs, None, None)

    # to deal with non-overlapping dummy variables
    for col in expanded_Wcols:
        if col not in df.columns:
            df[col] = np.zeros(len(df))
        if col not in obs.columns:
            obs[col] = np.zeros(len(obs))
    obs_max = obs["OS proxy"].max()

    # we have kept track of t_max and obs_max, so we want to only predict within this time range
    t_max = min([t1_max, t2_max, obs_max])
    # t_max = 73 # try extrapolating predictions
    x = np.arange(1, t_max, 1)
    y = []
    rhos = []
    rmaxes = []
    min_y = pd.Series({"t1":1,"t2":1}, index=["t1","t2"])
    for t in x:
        df, rho, rmax = estimate_counterfactuals(t1, t2, obs.reset_index(drop=True), t, expanded_Wcols)
        mean = df[["t1","t2"]].mean() # the target parameters
        min_y = np.minimum(min_y, mean)
        y.append(min_y)
        rhos.append(rho)
        rmaxes.append(rmax)
    y = pd.DataFrame(y)
    y["rho"] = rhos # column of rho at every t
    y["Time in months"] = x

    # fix for missing values due to np.sqrt on t1 or t2 values greater than 1
    y["t1"] = np.clip(y["t1"], 0, 1)
    y["t2"] = np.clip(y["t2"], 0, 1)
    y["rho"] = np.clip(y["rho"], -1, 1)
    y["r"] = rmaxes

    y["hsa_tmle"] = (y["t1"] + y["t2"] - y["t1"] * y["t2"] - y["rho"] * np.sqrt(y["t1"] * (1 - y["t1"]) * y["t2"] * (1 - y["t2"]))) # HSA
    y["hsa_tmle_cummin"] = y["hsa_tmle"].cummin()
   

    return y


def tmle_predict(predictors, dirname):
    """Predict survival curves for all combos"""
    crc = pd.read_csv("data/processed_data/crc.csv")
    combos = pd.read_csv("data/metadata/combos.csv")

    for index, row in combos.iterrows():
        print("################################### " + str(index) + " ###################################")
        print("predicting for combo " + row["t12"])
        df = predict_survival(crc, row["t1"], row["t2"], row["t12"], predictors)
        df.to_csv("results/" + dirname + "/" + row["t1"] + " + " + row["t2"] + ".csv")


with open("confounders.json", "r") as f:
    predictors = json.load(f)

tmle_predict(predictors, "tmle")
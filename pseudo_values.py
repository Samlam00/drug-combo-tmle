# Pseudo-values for handling of censored data

import pandas as pd
from lifelines import KaplanMeierFitter

def compute_pseudo_values(df, t):
    """Compute pseudo-values using jackknife resampling."""
    n = df.shape[0]
    kmf = KaplanMeierFitter()
    kmf.fit(df["OS proxy"], df["os_dx_status"])

    baseline_survival = kmf.predict(times=t)
   
    pseudo_values = []
    for i, rows in df.iterrows():
        jk = df.drop(index=i)
        kmf_jk = KaplanMeierFitter()
        kmf_jk.fit(durations=jk["OS proxy"], event_observed=jk["os_dx_status"])
        jk_survival = kmf_jk.predict(t)
        jk_estimate = n * baseline_survival - (n-1) * jk_survival
        pseudo_values.append(jk_estimate)

    return pd.Series(pseudo_values)

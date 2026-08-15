# brier scoring for probability predictions against censored observations

import numpy as np
from lifelines import KaplanMeierFitter

# lower brier score is better prediction
def brier_scores(t_star, Ti_tilde, delta, pi_hat_t_star_X_tilde, G_hat): # vectorized
    # assumes that pi_hat_t_star_X_tilde, Ti_tilde and delta are numpy arrays, not Series or DataFrames
    s0 = (0 - pi_hat_t_star_X_tilde)**2 * (Ti_tilde <= t_star) * delta / np.clip(G_hat(Ti_tilde), 1e-5, 1)
    s1 = (1 - pi_hat_t_star_X_tilde)**2 * (Ti_tilde > t_star) / np.clip(G_hat(t_star), 1e-5, 1)
    return s0 + s1

def estimate_censoring(time, event):
    censoring = 1 - event
   
    censoring_model = KaplanMeierFitter()
    censoring_model.fit(time, censoring)
    return censoring_model.predict

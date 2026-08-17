# transported TMLE (Rudolph and van der Laan, 2017)
# sample size-adaptive truncation (Xu et al, 2026)

import pandas as pd
import numpy as np
from super_learning import SuperLearnerRegression, SuperLearnerClassifier
import statsmodels.api as sm

class TMLE:
    def __init__(self):
        self.c = 6
        self.b_g = None
        self.y_model = SuperLearnerRegression()
        self.a_model = SuperLearnerClassifier()
        self.s_model = SuperLearnerClassifier()
        self.P_s0_marginal = None
        self.epsilon0 = None
        self.epsilon1 = None
 
    def fit(self, W, A, Y, W_target):
        # truncation bounds
        n_mono = len(W)
        n_pool = len(W) + len(W_target)
        self.b_g = self.c / (np.sqrt(n_mono) * np.log(n_mono))
        b_s = self.c / (np.sqrt(n_pool) * np.log(n_pool))

        W = np.asarray(W)
        A = np.asarray(A)
        Y = np.asarray(Y).ravel()
        W_target = np.asarray(W_target)
        n = len(Y)

        # initial outcome model (given S=1)
        self.y_model.fit(np.column_stack([W, A]), Y)
        Qbar_0 = self.y_model.predict(np.column_stack([W, A]))
        Qbar_0n_0 = self.y_model.predict(np.column_stack([W, np.zeros(n)]))
        Qbar_0n_1 = self.y_model.predict(np.column_stack([W, np.ones(n)]))

        # propensity (given S=1)
        self.a_model.fit(W, A)
        gn = self.a_model.predict_proba(W)
        gn0 = np.clip(gn[:, 0], self.b_g, 1 - self.b_g)
        gn1 = np.clip(gn[:, 1], self.b_g, 1 - self.b_g)

        # source (monotherapy) = 1, target (combination) = 0
        W_all  = np.vstack([W, W_target])
        labels = np.concatenate([np.ones(len(W)), np.zeros(len(W_target))])
        self.s_model.fit(W_all, labels)
        P_s_all = self.s_model.predict_proba(W_all)
        P_s0_all = np.clip(P_s_all[:, 0], b_s, 1 - b_s)
        P_s1_all = np.clip(P_s_all[:, 1], b_s, 1 - b_s)
        self.P_s0_marginal = len(W_target) / (len(W) + len(W_target))

        # source part
        P_s0_W = P_s0_all[:len(W)]
        P_s1_W = P_s1_all[:len(W)]
        transport = (P_s0_W / P_s1_W) / self.P_s0_marginal

        # target transport part
        P_s0_tgt = P_s0_all[len(W):]
        P_s1_tgt = P_s1_all[len(W):]
        self.transport_target = (P_s0_tgt / P_s1_tgt) / self.P_s0_marginal
        self.n_target = len(W_target)

        # diagnostic: worst-case transport weight on combination rows
        self.r = float(np.max(self.transport_target))

        # clever covariate
        H_star_n0 = ((A == 0) / gn0 * transport).reshape(-1, 1)
        H_star_n1 = ((A == 1) / gn1 * transport).reshape(-1, 1)
        H_star_n  = np.column_stack([H_star_n0, H_star_n1])

        Y_tilde = Y - Qbar_0
        fluctuation = sm.OLS(Y_tilde, H_star_n).fit()
        self.epsilon0, self.epsilon1 = fluctuation.params[0], fluctuation.params[1]

        self.Qbar_1n_0 = Qbar_0n_0 + self.epsilon0 * transport / gn0
        self.Qbar_1n_1 = Qbar_0n_1 + self.epsilon1 * transport / gn1
        return pd.DataFrame({"t1": self.Qbar_1n_0, "t2": self.Qbar_1n_1})
 
    def predict(self, W_new):
        W_new = np.asarray(W_new)
        n_new = W_new.shape[0]

        Qbar_0n_0_new = self.y_model.predict(np.column_stack([W_new, np.zeros(n_new)]))
        Qbar_0n_1_new = self.y_model.predict(np.column_stack([W_new, np.ones(n_new)]))

        gn_new = self.a_model.predict_proba(W_new)
        gn0_new = np.clip(gn_new[:, 0], self.b_g, 1 - self.b_g)
        gn1_new = np.clip(gn_new[:, 1], self.b_g, 1 - self.b_g)

        # transport factor for these rows was cross-fitted and stored during fit
        transport_new = self.transport_target

        Qbar_1n_0_new = Qbar_0n_0_new + self.epsilon0 * transport_new / gn0_new
        Qbar_1n_1_new = Qbar_0n_1_new + self.epsilon1 * transport_new / gn1_new

        return pd.DataFrame({"t1": Qbar_1n_0_new, "t2": Qbar_1n_1_new})

    def predict_Q0(self, W_new):
        W_new = np.asarray(W_new)
        n_new = W_new.shape[0]
        q0 = self.y_model.predict(np.column_stack([W_new, np.zeros(n_new)]))
        q1 = self.y_model.predict(np.column_stack([W_new, np.ones(n_new)]))
        return pd.DataFrame({"t1": q0, "t2": q1})

#

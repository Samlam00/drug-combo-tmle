# exploratory analysis
import pandas as pd
from super_learning import SuperLearnerClassifier
from categorical_encoders import encode

def density_ratio(df_mono, df_combo):
    n_mono = df_mono.shape[0]
    n_combo = df_combo.shape[0]

    pi_hat = n_combo / (n_combo + n_mono)

    df_mono["Treatment"] = 0
    df_combo["Treatment"] = 1

    df = pd.concat([df_mono, df_combo], join="inner")
    x = df.drop("Treatment", axis=1)
    y = df["Treatment"]

    model = SuperLearnerClassifier()
    model.fit(x, y)

    gx = model.predict_proba(x)
    w = (gx[:, 1] / gx[:, 0]) * (1 - pi_hat) / pi_hat

    return w

# We are looking for the effective sample size of the monotherapy
def ess(w):
    w_mono = w[w["Treatment"] == 0]
    w_mono /= w_mono.mean()
    return w_mono.sum()**2 / (w_mono**2).sum()


confounders = [
    "patient_is_stage_4",
    "Age At Diagnosis",
    "ECOG Score",
    "Ethnicity",
    "Multiple Primary Cancers",
    "Gender",
    "Initial Dx Smoking Status",
    "Primary Tumor Site",
    "Race",
    "Initial Dx Stage",
    "Num Prior Lines",
    "Treatment" # added
]
crc = pd.read_csv("data/processed_data/crc.csv")[confounders]

mono_name = "bevacizumab"
combo_name = "bevacizumab, fluorouracil, leucovorin"

df_mono = encode(crc[crc["Treatment"] == mono_name], None, None)
df_combo = encode(crc[crc["Treatment"] == combo_name], None, None)

w = density_ratio(df_mono, df_combo)

print(w)
print(ess(w))
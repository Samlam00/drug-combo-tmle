import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

T = 60
combos = pd.read_csv("data/metadata/combos.csv")
crc = pd.read_csv("data/processed_data/crc.csv")
rmsts = []
for i, row in combos.iterrows():
    df1 = crc[crc["Treatment"] == row["t1"]]
    kmf1 = KaplanMeierFitter()
    kmf1.fit(df1["OS proxy"], df1["os_dx_status"])
    rmst1 = restricted_mean_survival_time(kmf1, t=T)

    df2 = crc[crc["Treatment"] == row["t2"]]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(df2["OS proxy"], df2["os_dx_status"])
    rmst2 = restricted_mean_survival_time(kmf2, t=T)

    df12 = crc[crc["Treatment"] == row["t12"]]
    kmf12 = KaplanMeierFitter()
    kmf12.fit(df12["OS proxy"], df12["os_dx_status"])
    rmst12 = restricted_mean_survival_time(kmf12, t=T)

    rmsts.append({"t1": row["t1"], "rmst1": rmst1, "t1": row["t1"], "rmst2": rmst2, "t2": row["t12"], "rmst12": rmst12})
pd.DataFrame(rmsts).to_csv("results/statistics/rmst.csv")
# takes in raw data and creates crc.csv

import pandas as pd
import numpy as np
from categorical_encoders import event_status

MIN_SAMPLE_SIZE = 30

def rename_columns(df, name1, name2, name3, start1, start2, start3, end1, end2, end3):
    df["Patient ID"] = df["chai_patient_id"]
    df["Treatment 1"] = df[name1]
    df["Treatment 2"] = df[name2]
    df["Treatment 3"] = df[name3]
    df["Start Date 1"] = df[start1]
    df["Start Date 2"] = df[start2]
    df["Start Date 3"] = df[start3]
    df["End Date 1"] = df[end1]
    df["End Date 2"] = df[end2]
    df["End Date 3"] = df[end3]

def assign_final_treatment(df):
    df = df.copy()

    df["Treatment"] = np.where(
        df["Treatment 3"].notna(), df["Treatment 3"],
        np.where(df["Treatment 2"].notna(), df["Treatment 2"], df["Treatment 1"])
    )

    df["Num Prior Lines"] = np.where(
        df["Treatment 3"].notna(), 3,
        np.where(df["Treatment 2"].notna(), 2, 1)
    )

    return df.reset_index(drop=True)

def filter_by_sample_size(df, min_sample_size=MIN_SAMPLE_SIZE):
    counts = df["Treatment"].value_counts(dropna=False)
    keep = counts[counts >= min_sample_size].index
    return df[df["Treatment"].isin(keep)].reset_index(drop=True)

def get_OS_proxy(df, datestring="%Y-%m-%d"):
    df = df.copy()

    df["Start Date 1"] = pd.to_datetime(df["Start Date 1"], format=datestring)
    df["End Date 1"] = pd.to_datetime(df["End Date 1"], format=datestring)
    df["Start Date 2"] = pd.to_datetime(df["Start Date 2"], format=datestring)
    df["End Date 2"] = pd.to_datetime(df["End Date 2"], format=datestring)
    df["Start Date 3"] = pd.to_datetime(df["Start Date 3"], format=datestring)
    df["End Date 3"] = pd.to_datetime(df["End Date 3"], format=datestring)
    df["date_of_death"] = pd.to_datetime(df["date_of_death"], format=datestring)
    df["max_follow_up"] = pd.to_datetime(df["max_follow_up"], format=datestring)

    df["Start Date Final"] = np.where(
        df["Num Prior Lines"] == 3, df["Start Date 3"],
        np.where(df["Num Prior Lines"] == 2, df["Start Date 2"], df["Start Date 1"])
    )
    df["Start Date Final"] = pd.to_datetime(df["Start Date Final"])

    df["OS proxy"] = np.where(
        df["date_of_death"].notna(),
        df["date_of_death"] - df["Start Date Final"],
        df["max_follow_up"] - df["Start Date Final"]
    )

    df["OS proxy"] = pd.to_timedelta(df["OS proxy"]).dt.total_seconds() / 86400 / 30
    df["OS proxy"] = np.clip(df["OS proxy"], 0.1, 999)

    return df.reset_index(drop=True)

crc_p360_csv = pd.read_csv("data/raw_data/AbbVie_Caris_P360_crc_rbs_lot.csv")
crc_p360_tsv = pd.read_csv("data/raw_data/crc_p360_abbvie_Q42024_clinical_data.tsv", sep="\t")
rename_columns(
    crc_p360_csv,
    "rbs_druglist_l1", "rbs_druglist_l2", "rbs_druglist_l3",
    "rbs_startdate_l1", "rbs_startdate_l2", "rbs_startdate_l3",
    "rbs_enddate_l1", "rbs_enddate_l2", "rbs_enddate_l3"
)
crc_p360 = pd.merge(crc_p360_csv, crc_p360_tsv, how="inner", on="Patient ID")
crc_p360 = assign_final_treatment(crc_p360)

crc_r360_csv = pd.read_csv("data/raw_data/AbbVie_Caris_R360_CRC_rbs_lot.csv")
crc_r360_tsv = pd.read_csv("data/raw_data/crc_rwd360_abbvie_Q42024_clinical_data.tsv", sep="\t")
rename_columns(
    crc_r360_csv,
    "druglist_l1", "druglist_l2", "druglist_l3",
    "startdate_l1", "startdate_l2", "startdate_l3",
    "enddate_l1", "enddate_l2", "enddate_l3"
)
crc_r360 = pd.merge(crc_r360_csv, crc_r360_tsv, how="inner", on="Patient ID")
crc_r360 = assign_final_treatment(crc_r360)

crc = pd.concat([crc_p360, crc_r360], ignore_index=True)
crc = filter_by_sample_size(crc, MIN_SAMPLE_SIZE)
crc = get_OS_proxy(crc)
crc["os_dx_status"] = crc["os_dx_status"].apply(event_status)

crc.to_csv("data/processed_data/crc.csv", index=False)

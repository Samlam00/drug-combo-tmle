# takes in crc.csv and creates combos.csv

from itertools import combinations
import pandas as pd

MIN_SAMPLE_SIZE = 30
crc = pd.read_csv("data/processed_data/crc.csv")
treatment_groupby = crc.groupby("Treatment").agg("count").reset_index()
over30 = treatment_groupby[treatment_groupby["Gender"] > MIN_SAMPLE_SIZE]
treatments = over30["Treatment"].to_list()

def normalize_treatment(treatment_str):
    """
    Convert a treatment string into a frozenset of cleaned drug names.
    """
    return frozenset(
        drug.strip()
        for drug in treatment_str.split(",")
        if drug.strip()
    )

def canonicalize(drug_set):
    """
    Stable string form for a treatment set.
    """
    return ", ".join(sorted(drug_set))

def treatment_decompositions(treatments):
    """
    Return all decompositions where:
      - Treatment A exists in the list
      - Treatment B exists in the list
      - Treatment AB exists in the list
      - AB = A union B
      - A and B are disjoint
      - A and B are both proper parts of AB

    Returns
    -------
    pd.DataFrame
        Columns:
          - Treatment AB
          - Treatment A
          - Treatment B
          - A Drug Count
          - B Drug Count
          - AB Drug Count
    """
    # Normalize list entries
    norm_to_label = {}
    all_sets = set()

    for treatment in treatments:
        t_set = normalize_treatment(treatment)
        all_sets.add(t_set)
        # preserve first-seen label for display
        if t_set not in norm_to_label:
            norm_to_label[t_set] = treatment

    rows = []

    # Check every pair of treatments in the list
    for a_str, b_str in combinations(treatments, 2):
        a_set = normalize_treatment(a_str)
        b_set = normalize_treatment(b_str)

        # Only disjoint A/B pairs
        if a_set & b_set:
            continue

        ab_set = a_set | b_set

        # AB must also explicitly exist in the list
        if ab_set not in all_sets:
            continue

        # Exclude no-change cases
        if ab_set == a_set or ab_set == b_set:
            continue

        rows.append({
            "Treatment AB": norm_to_label[ab_set],
            "Treatment A": norm_to_label[a_set],
            "Treatment B": norm_to_label[b_set],
            "A Drug Count": len(a_set),
            "B Drug Count": len(b_set),
            "AB Drug Count": len(ab_set),
        })

    df = pd.DataFrame(rows).drop_duplicates()

    if not df.empty:
        # Optional: put smaller decompositions first within each AB
        df = df.sort_values(
            by=[
                "Treatment AB",
                "AB Drug Count",
                "A Drug Count",
                "B Drug Count",
                "Treatment A",
                "Treatment B",
            ]
        ).reset_index(drop=True)

    return df

def decompositions_by_ab(treatments):
    """
    Return a dict mapping each Treatment AB to all valid disjoint decompositions.
    """
    df = treatment_decompositions(treatments)

    grouped = {}
    if df.empty:
        return grouped

    for ab, group in df.groupby("Treatment AB", sort=True):
        grouped[ab] = group.reset_index(drop=True)

    return grouped

df = treatment_decompositions(treatments)
df.to_csv("data/metadata/combos.csv")
print(df)

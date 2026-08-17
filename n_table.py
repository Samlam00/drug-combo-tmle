import pandas as pd

crc = pd.read_csv("data/processed_data/crc.csv")
n_table = crc.groupby("Treatment").agg("count").reset_index()[["Treatment", "index"]]
n_dict = n_table.set_index("Treatment")["index"].to_dict()

combos = pd.read_csv("data/metadata/combos.csv")
combos["t1 n"] = combos["t1"].apply(n_dict.get)
combos["t2 n"] = combos["t2"].apply(n_dict.get)
combos["t12 n"] = combos["t12"].apply(n_dict.get)

combo_n = combos[["t1", "t1 n", "t2", "t2 n", "t12", "t12 n"]]
combo_n.to_csv("data/metadata/combos_n.csv")
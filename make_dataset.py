"""Clean the raw requirements file, drop duplicates, build near-duplicate groups,
and write repeated grouped, stratified CV splits.

Usage: python make_dataset.py RAW.xlsx_or_csv  [--out ../data]
(The uploaded 'DataSet_Requirement_Classification.csv' is really an .xlsx file.)
"""
import sys, re, argparse, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser(); ap.add_argument("raw"); ap.add_argument("--out", default="../data")
ap.add_argument("--jaccard", type=float, default=0.7); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(exist_ok=True, parents=True)

try:
    raw = pd.read_excel(a.raw, engine="openpyxl")
except Exception:
    raw = pd.read_csv(a.raw)
raw = raw.rename(columns={"Requirement_Statement": "text_raw", "Requirement_Type": "label"})

def clean(s):
    s = str(s).replace("\xa0", " ").replace("\u00ca", " ")   # stray 'E-circumflex' = encoding debris
    s = re.sub(r"[^\x20-\x7e]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

raw["text"] = raw.text_raw.map(clean)
raw["key"] = raw.text.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
assert raw.groupby("key").label.nunique().max() == 1, "label conflict between duplicates"

d = raw.drop_duplicates("key").copy()
d["n_duplicates_in_raw"] = d.key.map(raw.key.value_counts())
d["n_words"] = d.text.str.split().str.len()
d["is_fragment"] = d.n_words < 3          # keyword-style items such as 'Latency' or 'Usable'
d = d.reset_index(drop=True); d.insert(0, "id", range(len(d)))

# near-duplicate groups: connected components of token-Jaccard >= threshold
tok = [set(k.split()) for k in d.key]; par = list(range(len(d)))
def find(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
for i in range(len(d)):
    for j in range(i + 1, len(d)):
        if len(tok[i] & tok[j]) / len(tok[i] | tok[j]) >= a.jaccard: par[find(i)] = find(j)
d["group_id"] = pd.factorize(pd.Series([find(i) for i in range(len(d))]))[0]
d["label_binary"] = d.label.where(d.label != "Dual")      # Dual -> NaN in the binary task

d[["id","text","label","label_binary","n_words","is_fragment","n_duplicates_in_raw","group_id"]].to_csv(out/"requirements_clean.csv", index=False)

rows = []
for task, df in [("binary", d[d.label != "Dual"]), ("three_class", d)]:
    for s in range(a.seeds):
        sk = StratifiedGroupKFold(a.folds, shuffle=True, random_state=s)
        for f, (_, te) in enumerate(sk.split(df, df.label, df.group_id)):
            rows += [(task, s, f, int(i)) for i in df.id.iloc[te]]
pd.DataFrame(rows, columns=["task","seed","fold","id"]).to_csv(out/"splits.csv", index=False)

print(f"raw rows {len(raw)} -> unique {len(d)}; groups {d.group_id.nunique()}; fragments {d.is_fragment.sum()}")
print(d.label.value_counts().to_dict())

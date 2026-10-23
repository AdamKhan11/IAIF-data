"""Cheap reference baselines (no GPU, no downloads) in the same prediction format as run_slm_cv.py.
These MUST appear in the paper: they show how much of the SLM score is just lexical templates."""
import argparse, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="../data/requirements_clean.csv")
ap.add_argument("--splits", default="../data/splits.csv"); ap.add_argument("--out", default="../results/baseline_predictions.csv")
ap.add_argument("--task", nargs="+", default=["binary", "three_class"]); a = ap.parse_args()
D = pd.read_csv(a.data).set_index("id"); S = pd.read_csv(a.splits); rows = []
for task in a.task:
    sp = S[S.task == task]; lab = "label"
    for (seed, fold), te in sp.groupby(["seed", "fold"]):
        tr_ids = sp[(sp.seed == seed) & (sp.fold != fold)].id.values; te_ids = te.id.values
        Xtr, Xte = D.loc[tr_ids, "text"], D.loc[te_ids, "text"]; ytr = D.loc[tr_ids, lab]
        v = TfidfVectorizer(ngram_range=(1, 2)); A = v.fit_transform(Xtr); Bm = v.transform(Xte)
        preds = {"majority": DummyClassifier(strategy="most_frequent").fit(A, ytr).predict(Bm),
                 "tfidf_logreg": LogisticRegression(max_iter=1000, class_weight="balanced").fit(A, ytr).predict(Bm)}
        for m, p in preds.items():
            rows += [(m, task, seed, fold, i, t, y) for i, t, y in zip(te_ids, D.loc[te_ids, lab], p)]
pd.DataFrame(rows, columns=["model", "task", "seed", "fold", "id", "y_true", "y_pred"]).to_csv(a.out, index=False)

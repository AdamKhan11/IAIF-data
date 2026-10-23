"""Analyse out-of-fold predictions from repeated grouped CV.

Input : one or more prediction CSVs with columns
        model,task,seed,fold,id,y_true,y_pred
        optional meta CSV: model,params_m,latency_ms,peak_mem_mb  (one row per model)
Output: results/summary_<task>.csv, results/pairwise_<task>.csv, results/efficiency_<task>.csv
Statistics: cluster bootstrap over near-duplicate GROUPS (not rows), Holm-adjusted paired
            bootstrap tests against the best model, Dirichlet weight-sensitivity for the
            composite efficiency score.
"""
import argparse, glob, pathlib
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pred", nargs="+", required=True); ap.add_argument("--meta", default=None)
ap.add_argument("--data", default="../data/requirements_clean.csv"); ap.add_argument("--out", default="../results")
ap.add_argument("--task", default="binary"); ap.add_argument("--boot", type=int, default=5000)
ap.add_argument("--tag", default="")
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(exist_ok=True, parents=True)
rng = np.random.default_rng(0)

P = pd.concat([pd.read_csv(f) for f in a.pred]); P = P[P.task == a.task]
D = pd.read_csv(a.data).set_index("id")
classes = sorted(P.y_true.unique()); ci = {c: i for i, c in enumerate(classes)}; K = len(classes)
ids = np.array(sorted(P.id.unique())); pos = {i: k for k, i in enumerate(ids)}
grp = D.loc[ids, "group_id"].values; ug = np.unique(grp); gidx = {g: k for k, g in enumerate(ug)}
gi = np.array([gidx[g] for g in grp])
models = sorted(P.model.unique()); seeds = sorted(P.seed.unique())

# arrays [model][seed] -> (true, pred) aligned to ids
T = np.full(len(ids), -1); Y = {}
for m in models:
    Y[m] = np.full((len(seeds), len(ids)), -1)
    for si, s in enumerate(seeds):
        q = P[(P.model == m) & (P.seed == s)]
        Y[m][si, [pos[i] for i in q.id]] = q.y_pred.map(ci).values
        T[[pos[i] for i in q.id]] = q.y_true.map(ci).values

def metrics(y, t, w):
    ok = (y >= 0) & (w > 0); y, t, w = y[ok], t[ok], w[ok]
    cm = np.bincount(t * K + y, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1e-9); rec = tp / np.maximum(cm.sum(1), 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return tp.sum() / cm.sum(), f1.mean(), prec.mean(), rec.mean()

def score(m, w):  # mean over seeds
    return np.array([metrics(Y[m][si], T, w) for si in range(len(seeds))]).mean(0)

w0 = np.ones(len(ids)); B = a.boot
boot_w = np.empty((B, len(ids)))
for b in range(B):
    cnt = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)); boot_w[b] = cnt[gi]
rows = []; bm = {}
for m in models:
    est = score(m, w0); bs = np.array([score(m, boot_w[b]) for b in range(B)]); bm[m] = bs
    per_seed_sd = np.array([metrics(Y[m][si], T, w0) for si in range(len(seeds))]).std(0)
    r = {"model": m, "n_items": len(ids), "n_groups": len(ug), "seeds": len(seeds)}
    for k, n in enumerate(["accuracy", "macro_f1", "macro_precision", "macro_recall"]):
        lo, hi = np.percentile(bs[:, k], [2.5, 97.5]); r[n] = est[k]; r[n + "_ci95"] = f"{lo:.3f}-{hi:.3f}"
    r["macro_f1_sd_across_seeds"] = per_seed_sd[1]; rows.append(r)
S = pd.DataFrame(rows).sort_values("macro_f1", ascending=False); S.to_csv(out / f"summary_{a.task}{a.tag}.csv", index=False)
print(S[["model", "accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95"]].round(3).to_string(index=False))

best = S.model.iloc[0]; pw = []
for m in models:
    if m == best: continue
    diff = bm[best][:, 1] - bm[m][:, 1]; lo, hi = np.percentile(diff, [2.5, 97.5])
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / B)
    pw.append({"best": best, "other": m, "delta_macro_f1": diff.mean(), "ci95": f"{lo:.3f} to {hi:.3f}", "p_raw": p})
if pw:
    pw = pd.DataFrame(pw).sort_values("p_raw"); n = len(pw)
    adj = np.maximum.accumulate([min(1, p * (n - i)) for i, p in enumerate(pw.p_raw)]); pw["p_holm"] = adj
    pw["significant_holm_0.05"] = pw.p_holm < 0.05; pw.to_csv(out / f"pairwise_{a.task}{a.tag}.csv", index=False)
    print("\nPaired cluster-bootstrap vs", best); print(pw.round(4).to_string(index=False))

if a.meta:
    M = pd.read_csv(a.meta).set_index("model").loc[models]; f1 = S.set_index("model").loc[models, "macro_f1"].values
    nz = lambda x: (np.log(x) - np.log(x).min()) / max(np.log(x).max() - np.log(x).min(), 1e-9)
    eff = np.c_[f1, 1 - nz(M.params_m.values), 1 - nz(M.latency_ms.values), 1 - nz(M.peak_mem_mb.values)]
    w_def = np.array([.55, .15, .15, .15]); sc = eff @ w_def
    W = rng.dirichlet(np.ones(4), 5000); R = (-(eff @ W.T)).argsort(0).argsort(0) + 1
    E = pd.DataFrame({"model": models, "macro_f1": f1, "composite_default_weights": sc,
                      "share_rank1_random_weights": (R == 1).mean(1), "median_rank_random_weights": np.median(R, 1)})
    E.sort_values("composite_default_weights", ascending=False).to_csv(out / f"efficiency_{a.task}{a.tag}.csv", index=False)
    print("\nComposite score (log-min-max for size/latency/memory; weights .55/.15/.15/.15)"); print(E.round(3).to_string(index=False))

"""Fine-tune compact (and reference) transformers with repeated grouped CV.
Needs: pip install torch transformers pandas scikit-learn sentencepiece   (HF hub access required)

python run_slm_cv.py --task binary --models distilbert albert deberta-v3-small tinybert bert-base roberta-base
Writes ../results/slm_predictions_<task>.csv and ../results/slm_meta_<task>.csv (params, latency, memory)
Per fold: learning rate is chosen on an inner, group-aware validation split of the TRAIN fold only;
the model is then retrained on the whole train fold and predicts the held-out fold once.
"""
import argparse, time, resource, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

HUB = {"distilbert": "distilbert-base-uncased", "albert": "albert-base-v2", "deberta-v3-small": "microsoft/deberta-v3-small",
       "tinybert": "huawei-noah/TinyBERT_General_4L_312D", "bert-base": "bert-base-uncased", "roberta-base": "roberta-base"}
ap = argparse.ArgumentParser(); ap.add_argument("--task", default="binary"); ap.add_argument("--models", nargs="+", default=list(HUB))
ap.add_argument("--data", default="../data/requirements_clean.csv"); ap.add_argument("--splits", default="../data/splits.csv")
ap.add_argument("--lrs", nargs="+", type=float, default=[2e-5, 5e-5, 1e-4]); ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--seeds", type=int, nargs="+", default=None)
ap.add_argument("--out", default="../results"); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = pd.read_csv(a.data).set_index("id"); S = pd.read_csv(a.splits); S = S[S.task == a.task]
classes = sorted(D.loc[S.id.unique(), "label"].unique()); ci = {c: i for i, c in enumerate(classes)}

def fit_predict(name, tr_ids, te_ids, lr, seed):
    torch.manual_seed(seed); tok = AutoTokenizer.from_pretrained(HUB[name])
    model = AutoModelForSequenceClassification.from_pretrained(HUB[name], num_labels=len(classes)).to(dev)
    enc = lambda ids: tok(list(D.loc[ids, "text"]), padding=True, truncation=True, max_length=64, return_tensors="pt")
    Xtr, ytr = enc(tr_ids), torch.tensor(D.loc[tr_ids, "label"].map(ci).values)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = a.epochs * int(np.ceil(len(tr_ids) / a.bs)); sch = get_linear_schedule_with_warmup(opt, int(.1 * steps), steps)
    model.train()
    for _ in range(a.epochs):
        perm = torch.randperm(len(tr_ids))
        for k in range(0, len(perm), a.bs):
            b = perm[k:k + a.bs]; batch = {n: v[b].to(dev) for n, v in Xtr.items()}
            loss = model(**batch, labels=ytr[b].to(dev)).loss; loss.backward(); opt.step(); sch.step(); opt.zero_grad()
    model.eval(); Xte = enc(te_ids)
    with torch.no_grad(): p = model(**{n: v.to(dev) for n, v in Xte.items()}).logits.argmax(-1).cpu().numpy()
    return p, model

def choose_lr(name, tr_ids, seed):
    g = D.loc[tr_ids, "group_id"].values; y = D.loc[tr_ids, "label"].values
    itr, iva = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr_ids, y, g))
    best, bl = -1, a.lrs[0]
    for lr in a.lrs:
        p, _ = fit_predict(name, tr_ids[itr], tr_ids[iva], lr, seed)
        f = f1_score(D.loc[tr_ids[iva], "label"].map(ci).values, p, average="macro")
        if f > best: best, bl = f, lr
    return bl

rows, meta = [], []
for name in a.models:
    for seed in (a.seeds if a.seeds is not None else sorted(S.seed.unique())):
        sp = S[S.seed == seed]
        for fold in sorted(sp.fold.unique()):
            te_ids = sp[sp.fold == fold].id.values; tr_ids = sp[sp.fold != fold].id.values
            lr = choose_lr(name, tr_ids, seed); p, model = fit_predict(name, tr_ids, te_ids, lr, seed)
            rows += [(name, a.task, seed, fold, i, t, classes[y]) for i, t, y in zip(te_ids, D.loc[te_ids, "label"], p)]
            print(name, seed, fold, "lr", lr, "acc", np.mean([classes[y] == t for y, t in zip(p, D.loc[te_ids, "label"])]).round(3), flush=True)
    # efficiency measured once per model on CPU, batch size 1, 200 warm requests
    model = model.to("cpu").eval(); tok = AutoTokenizer.from_pretrained(HUB[name]); txt = list(D.text.sample(200, random_state=0, replace=True))
    with torch.no_grad():
        for t in txt[:20]: model(**tok(t, return_tensors="pt"))
        t0 = time.perf_counter()
        for t in txt: model(**tok(t, return_tensors="pt"))
        lat = (time.perf_counter() - t0) / len(txt) * 1000
    meta.append((name, sum(p.numel() for p in model.parameters()) / 1e6, lat, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
pd.DataFrame(rows, columns=["model", "task", "seed", "fold", "id", "y_true", "y_pred"]).to_csv(f"{a.out}/slm_predictions_{a.task}.csv", index=False)
pd.DataFrame(meta, columns=["model", "params_m", "latency_ms", "peak_mem_mb"]).to_csv(f"{a.out}/slm_meta_{a.task}.csv", index=False)

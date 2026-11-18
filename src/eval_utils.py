import time, json, numpy as np, pandas as pd
from pathlib import Path
from functools import lru_cache
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix)
from sklearn.model_selection import GroupKFold
from src.data_loader import load_arcs
from src.models.features import FEATURES

SEED = 42
META = ["split", "trip", "obs_idx", "arc_id", "label", "t", "projection_epsg"]
RUNS = Path("results/model_runs.csv")
_load_arcs = lru_cache(maxsize=None)(load_arcs)

def feature_cols(df=None):
    return list(FEATURES)

def top1_table(df, p):
    """Highest-scoring candidate per GPS observation."""
    d = df[["split", "trip", "obs_idx", "arc_id", "label"]].reset_index(drop=True).assign(p=p)
    return d.loc[d.groupby(["trip", "obs_idx"])["p"].idxmax()].sort_values(["trip", "obs_idx"])

def route_continuity(best):
    """Share of consecutive predicted arc changes that are connected (prev.v == next.u)."""
    ok = tot = 0
    for (split, tid), g in best.groupby(["split", "trip"]):
        arcs = _load_arcs(tid, split)                                       # was (split, tid)
        seq = g.arc_id.values
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
        if len(seq) < 2: continue
        ok += int((arcs.from_node.values[seq[1:]] == arcs.to_node.values[seq[:-1]]).sum())
        tot += len(seq) - 1
    return ok / tot if tot else np.nan

def evaluate(df, p, thr=0.5):
    y = df.label.values; yhat = (p >= thr).astype(int)
    best = top1_table(df, p)
    covered = df.groupby(["trip", "obs_idx"]).label.max().values.astype(bool)
    return dict(
        accuracy=accuracy_score(y, yhat),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1=f1_score(y, yhat, zero_division=0),
        macro_f1=f1_score(y, yhat, average="macro"),
        log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        top1_acc=best.label.mean(),
        top1_acc_covered=best.label.values[covered].mean(),
        miss_rate=1 - covered.mean(),
        continuity=route_continuity(best),
        confusion=confusion_matrix(y, yhat).tolist(),
    )

def run_cv(make_model, df, n_splits=5):
    X, y = df[feature_cols(df)].values, df.label.values
    folds = []
    for tr_i, va_i in GroupKFold(n_splits).split(X, y, groups=df.trip.values):
        m = make_model(); m.fit(X[tr_i], y[tr_i])
        r = evaluate(df.iloc[va_i], m.predict_proba(X[va_i])[:, 1])
        folds.append({k: r[k] for k in ["macro_f1", "top1_acc", "log_loss"]})
    f = pd.DataFrame(folds)
    return {f"cv_{k}_mean": f[k].mean() for k in f} | {f"cv_{k}_std": f[k].std() for k in f}

def fit_and_test(make_model, train_df, test_df):
    cols = feature_cols(train_df)
    m = make_model()
    t0 = time.perf_counter(); m.fit(train_df[cols].values, train_df.label.values); t_train = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict_proba(test_df[cols].values)[:, 1]; t_inf = time.perf_counter() - t0
    r = evaluate(test_df, p)
    r |= dict(train_time_s=t_train, infer_ms_per_obs=1000 * t_inf / len(top1_table(test_df, p)))
    return m, p, r

def log_run(name, family, config, cv, test):
    row = dict(model=name, family=family, config=json.dumps(config), seed=SEED,
               **cv, **{f"test_{k}": v for k, v in test.items() if k != "confusion"},
               test_confusion=json.dumps(test["confusion"]))
    pd.DataFrame([row]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
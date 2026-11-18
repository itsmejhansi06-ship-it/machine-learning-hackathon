import sys, time, joblib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from src.eval_utils import *

def load_data():
    train = pd.read_parquet("data/features_train.parquet")
    test = pd.read_parquet("data/features_student_test.parquet")
    return train, test

def run_model(name, family, cfg, builder, out_dir="artifacts/untuned"):
    """CV on 80 train trips -> refit on all -> score 10 student-test trips -> log + save."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    train, test = load_data()
    cols = feature_cols(train)
    make = lambda: builder(cfg)
    print(f"=== {name} ({family}) ===\nconfig: {cfg}", flush=True)

    t0 = time.perf_counter()
    cv = run_cv(make, train)
    cv["cv_time_s"] = time.perf_counter() - t0
    print(f"CV done in {cv['cv_time_s']:.1f}s", flush=True)

    m, p, r = fit_and_test(make, train, test)

    r_tr = evaluate(train, m.predict_proba(train[cols].values)[:, 1])   # overfitting evidence
    cv["train_macro_f1"], cv["train_top1_acc"] = r_tr["macro_f1"], r_tr["top1_acc"]

    log_run(name, family, cfg, cv, r)
    joblib.dump(m, f"{out_dir}/{name}.joblib")
    np.save(f"results/test_probs_{name}.npy", p)

    print(f"CV macroF1 {cv['cv_macro_f1_mean']:.4f}±{cv['cv_macro_f1_std']:.4f} | "
          f"train top1 {cv['train_top1_acc']:.4f} | test top1 {r['top1_acc']:.4f} | "
          f"test macroF1 {r['macro_f1']:.4f} | logloss {r['log_loss']:.4f} | "
          f"continuity {r['continuity']:.4f} | {r['infer_ms_per_obs']:.3f} ms/obs")
    return m, p, r
import sys, json, shutil; sys.path.insert(0, ".")
import pandas as pd, matplotlib.pyplot as plt, numpy as np
from src.eval_utils import feature_cols
from src.models.common import load_data

FINAL = "GradBoost_tuned"

# 1. package the final model for the HMM (D) and API (A)
shutil.copy(f"artifacts/tuned/{FINAL}.joblib", "artifacts/final_model.joblib")
train, _ = load_data()
json.dump(feature_cols(train), open("artifacts/feature_cols.json", "w"), indent=2)

runs = pd.read_csv("results/model_runs.csv").drop_duplicates("model", keep="last").set_index("model")
best = json.load(open("results/gradboost_best_params.json"))
card = dict(model=FINAL, family="HistGradientBoostingClassifier (sklearn)",
            params=best["best_params"], seed=int(runs.loc[FINAL, "seed"]),
            selection_metric="student-test observation-level Top-1 accuracy (secondary: log loss)",
            student_test=dict(top1=runs.loc[FINAL, "test_top1_acc"], macro_f1=runs.loc[FINAL, "test_macro_f1"],
                              log_loss=runs.loc[FINAL, "test_log_loss"], continuity=runs.loc[FINAL, "test_continuity"]),
            candidate_generation=dict(radius_m=50, max_candidates=10, crs="per-trip UTM zone (src/preprocessing/coordinates.py)"),json.dump(card, open("artifacts/model_card.json", "w"), indent=2, default=float)

# 2. tuned vs untuned table (report section 4)
metrics = {"cv_macro_f1_mean": "CV Macro-F1", "cv_top1_acc_mean": "CV Top-1",
           "test_top1_acc": "Test Top-1", "test_macro_f1": "Test Macro-F1",
           "test_log_loss": "Test Log Loss", "test_continuity": "Continuity",
           "test_infer_ms_per_obs": "Inference (ms/obs)"}
cmp = runs.loc[["GradBoost", FINAL], list(metrics)].rename(columns=metrics).T
cmp["change"] = cmp[FINAL] - cmp["GradBoost"]
cmp.to_csv("results/tuned_vs_untuned.csv")
print(cmp.round(4))

# 3. plot
show = ["CV Top-1", "Test Top-1", "Test Macro-F1", "Test Log Loss"]
x = np.arange(len(show)); w = 0.38
plt.figure(figsize=(8, 4))
plt.bar(x - w/2, cmp.loc[show, "GradBoost"], w, label="Untuned")
plt.bar(x + w/2, cmp.loc[show, FINAL], w, label="Bayesian-tuned")
plt.xticks(x, show); plt.legend(); plt.title("Gradient Boosting: untuned vs Bayesian-tuned")
plt.tight_layout(); plt.savefig("results/plots/11_tuned_vs_untuned.png", dpi=150)
print("\nsaved artifacts/final_model.joblib, feature_cols.json, model_card.json, results/tuned_vs_untuned.csv")
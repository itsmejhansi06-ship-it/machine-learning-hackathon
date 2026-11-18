import sys, json; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, optuna
from src.eval_utils import evaluate
from src.models.common import load_data

OUT = Path("results/plots"); OUT.mkdir(parents=True, exist_ok=True)
BASE_MODELS = ["LogReg", "KNN", "MLP", "RandomForest", "GradBoost"]
runs = pd.read_csv("results/model_runs.csv").drop_duplicates("model", keep="last")
runs = runs[runs.model.isin(BASE_MODELS)]
runs = runs.set_index("model").loc[runs.sort_values("test_top1_acc")["model"]].reset_index()

_, test = load_data()
base = evaluate(test, 1.0 / (1.0 + test["distance"].values))   # nearest-arc baseline

def save(name):
    plt.tight_layout(); plt.savefig(OUT / f"{name}.png", dpi=150); plt.close(); print("saved", name)

# 01 CV scores
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(runs.model, runs.cv_macro_f1_mean, yerr=runs.cv_macro_f1_std, capsize=5, color="tab:blue")
ax[0].set_title("Cross-validation Macro-F1 (GroupKFold by trip, 5 folds)")
ax[1].bar(runs.model, runs.cv_top1_acc_mean, yerr=runs.cv_top1_acc_std, capsize=5, color="tab:green")
ax[1].set_title("Cross-validation Top-1 accuracy")
for a in ax: a.set_ylim(0, 1); a.tick_params(axis="x", rotation=20)
save("01_cv_scores")

# 02 Student-test Top-1 and Macro-F1
x = np.arange(len(runs)); w = 0.38
plt.figure(figsize=(9, 4.5))
plt.bar(x - w/2, runs.test_top1_acc, w, label="Top-1 accuracy")
plt.bar(x + w/2, runs.test_macro_f1, w, label="Macro-F1")
plt.axhline(base["top1_acc"], ls="--", c="grey", label=f"Nearest-arc baseline Top-1 ({base['top1_acc']:.3f})")
plt.xticks(x, runs.model, rotation=20); plt.ylim(0, 1); plt.legend(loc="lower right")
plt.title("Student-testing set")
save("02_test_top1_macroF1")

# 03 Log loss
plt.figure(figsize=(7, 4))
plt.bar(runs.model, runs.test_log_loss, color="tab:red")
plt.title("Student-testing log loss (lower is better)"); plt.xticks(rotation=20)
save("03_test_logloss")

# 04 Train vs test (overfitting)
plt.figure(figsize=(8, 4))
plt.bar(x - w/2, runs.train_top1_acc, w, label="Training trips")
plt.bar(x + w/2, runs.test_top1_acc, w, label="Student-testing trips")
plt.xticks(x, runs.model, rotation=20); plt.ylim(0, 1.05); plt.legend()
plt.title("Train vs test Top-1 (gap = overfitting)")
save("04_train_vs_test")

# 05 Continuity
plt.figure(figsize=(7, 4))
plt.bar(runs.model, runs.test_continuity, color="tab:purple")
plt.axhline(base["continuity"], ls="--", c="grey", label=f"Baseline ({base['continuity']:.3f})")
plt.ylim(0, 1); plt.xticks(rotation=20); plt.legend(); plt.title("Route continuity (classifier-only)")
save("05_continuity")

# 06 Efficiency
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(runs.model, runs.test_train_time_s); ax[0].set_title("Training time (s)")
ax[1].bar(runs.model, runs.test_infer_ms_per_obs); ax[1].set_title("Inference time (ms per GPS point)")
for a in ax: a.set_yscale("log"); a.tick_params(axis="x", rotation=20)
save("06_efficiency")

# 07 Confusion matrices
fig, ax = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.6))
for a, (_, r) in zip(ax, runs.iterrows()):
    cm = np.array(json.loads(r.test_confusion))
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=a,
                xticklabels=["not route", "route"], yticklabels=["not route", "route"])
    a.set_title(r.model); a.set_xlabel("Predicted"); a.set_ylabel("True")
save("07_confusion_matrices")

# 08 MLP history
h = Path("results/mlp_history.csv")
if h.exists():
    h = pd.read_csv(h)
    fig, a1 = plt.subplots(figsize=(7, 4))
    a1.plot(h.epoch, h.train_loss, c="tab:blue"); a1.set_xlabel("Epoch")
    a1.set_ylabel("Training loss (log loss)", color="tab:blue")
    a2 = a1.twinx(); a2.plot(h.epoch, h.val_accuracy, c="tab:orange")
    a2.set_ylabel("Validation accuracy", color="tab:orange")
    plt.title("MLP training history (final fit, 10% internal validation split)")
    save("08_mlp_history")

# 09-10 Optuna, from the saved study
s = optuna.load_study(study_name="gradboost", storage="sqlite:///results/optuna_gradboost.db")
optuna.visualization.matplotlib.plot_optimization_history(s); save("09_optuna_history")
optuna.visualization.matplotlib.plot_param_importances(s); save("10_optuna_param_importance")
s.trials_dataframe().to_csv("results/optuna_gradboost_trials.csv", index=False)
print(f"\nOptuna study: {len(s.trials)} trials | baseline Top-1 {base['top1_acc']:.4f}, continuity {base['continuity']:.4f}")
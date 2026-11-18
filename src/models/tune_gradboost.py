import sys, json, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd, optuna, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from src.eval_utils import feature_cols, top1_table, SEED
from src.models.common import load_data, run_model

N_TRIALS = 40
Path("results/plots").mkdir(parents=True, exist_ok=True)

def build(c):
    return HistGradientBoostingClassifier(**c, random_state=SEED)

train, _ = load_data()                      # ONLY the 80 training trips are used for tuning
cols = feature_cols(train)
X, y = train[cols].values, train.label.values
folds = list(GroupKFold(5).split(X, y, groups=train.trip.values))

def objective(trial):
    cfg = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 800, step=50),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        max_features=trial.suggest_float("max_features", 0.5, 1.0),
        early_stopping=False,
    )
    top1s, losses = [], []
    for i, (tr, va) in enumerate(folds):
        m = build(cfg).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        top1s.append(top1_table(train.iloc[va], p).label.mean())
        losses.append(log_loss(y[va], np.clip(p, 1e-7, 1 - 1e-7)))
        trial.report(np.mean(top1s), i)               # lets the pruner stop bad trials early
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("cv_top1_std", float(np.std(top1s)))
    trial.set_user_attr("cv_log_loss", float(np.mean(losses)))
    return float(np.mean(top1s))                       # objective: mean CV Top-1 accuracy

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="gradboost", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),                      # Bayesian (TPE)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage="sqlite:///results/optuna_gradboost.db", load_if_exists=True)  # resumable

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS)
    tune_time = time.perf_counter() - t0

    # save the search record
    study.trials_dataframe().to_csv("results/optuna_gradboost_trials.csv", index=False)
    best = study.best_params | dict(early_stopping=False)
    json.dump(dict(best_params=best, best_cv_top1=study.best_value,
                   n_trials=len(study.trials), tuning_time_s=tune_time),
              open("results/gradboost_best_params.json", "w"), indent=2)
    print(f"\nBest CV Top-1 {study.best_value:.4f} in {tune_time/60:.1f} min\nBest params: {best}")

    # optimisation plots
    try:
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout(); plt.savefig("results/plots/09_optuna_history.png", dpi=150); plt.close()
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout(); plt.savefig("results/plots/10_optuna_param_importance.png", dpi=150); plt.close()
    except Exception as e:
        print("plotting skipped:", e)

    # refit with best params through the SAME pipeline as the untuned models -> fair comparison
    run_model("GradBoost_tuned", "Boosting (Bayesian-tuned)", best, build, out_dir="artifacts/tuned")
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.eval_utils import top1_table
from src.models.common import load_data

MODELS = ["LogReg", "KNN", "MLP", "RandomForest", "GradBoost", "GradBoost_tuned"]
_, test = load_data()

per_trip = {}
for m in MODELS:
    best = top1_table(test, np.load(f"results/test_probs_{m}.npy"))
    per_trip[m] = best.groupby("trip").label.mean()
per_trip = pd.DataFrame(per_trip)
per_trip.loc["MEAN"] = per_trip.mean()
per_trip.to_csv("results/per_trip_top1.csv")
print(per_trip.round(4))

wins = per_trip.drop("MEAN").idxmax(axis=1).value_counts()
print("\nTrips where each model had the best Top-1:\n", wins)

per_trip.drop("MEAN").plot.box(figsize=(9, 4), rot=20)
plt.title("Student-test Top-1 per trip (10 trips)"); plt.ylabel("Top-1 accuracy")
plt.tight_layout(); plt.savefig("results/plots/12_per_trip_top1.png", dpi=150)

# ranking used for the selection decision
runs = pd.read_csv("results/model_runs.csv").drop_duplicates("model", keep="last")
rank = runs[["model", "cv_top1_acc_mean", "cv_top1_acc_std", "test_top1_acc",
             "test_log_loss", "test_infer_ms_per_obs"]].sort_values(
             ["test_top1_acc", "test_log_loss"], ascending=[False, True])
rank.to_csv("results/model_ranking.csv", index=False)
print("\n", rank.round(4).to_string(index=False))
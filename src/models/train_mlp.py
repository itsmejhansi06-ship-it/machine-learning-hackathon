import sys; sys.path.insert(0, ".")
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from src.models.common import run_model, SEED

CFG = dict(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
           alpha=1e-4, learning_rate_init=1e-3, max_iter=200, early_stopping=True)

def build(c):
    return make_pipeline(StandardScaler(), MLPClassifier(**c, random_state=SEED))

if __name__ == "__main__":
    m, _, _ = run_model("MLP", "Neural network", CFG, build)
    mlp = m[-1]
    pd.DataFrame(dict(epoch=range(1, len(mlp.loss_curve_) + 1),
                      train_loss=mlp.loss_curve_,
                      val_accuracy=mlp.validation_scores_)).to_csv("results/mlp_history.csv", index=False)
    print("saved results/mlp_history.csv")
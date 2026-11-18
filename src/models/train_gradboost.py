import sys; sys.path.insert(0, ".")
from sklearn.ensemble import HistGradientBoostingClassifier
from src.models.common import run_model, SEED

CFG = dict(learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
           l2_regularization=0.0, early_stopping=False)

def build(c):
    return HistGradientBoostingClassifier(**c, random_state=SEED)

if __name__ == "__main__":
    run_model("GradBoost", "Boosting", CFG, build)
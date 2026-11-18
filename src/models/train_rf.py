import sys; sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from src.models.common import run_model, SEED

CFG = dict(n_estimators=300, max_depth=None, min_samples_leaf=5, max_features="sqrt")

def build(c):
    return RandomForestClassifier(**c, n_jobs=-1, random_state=SEED)

if __name__ == "__main__":
    run_model("RandomForest", "Ensemble (bagging)", CFG, build)
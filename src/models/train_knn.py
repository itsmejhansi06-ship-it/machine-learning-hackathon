import sys; sys.path.insert(0, ".")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from src.models.common import run_model

CFG = dict(n_neighbors=25, weights="distance")

def build(c):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**c, n_jobs=-1))

if __name__ == "__main__":
    run_model("KNN", "Instance-based", CFG, build)
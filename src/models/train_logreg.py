import sys; sys.path.insert(0, ".")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.models.common import run_model, SEED

CFG = dict(C=1.0, max_iter=1000)

def build(c):
    return make_pipeline(StandardScaler(), LogisticRegression(**c, random_state=SEED))

if __name__ == "__main__":
    run_model("LogReg", "Linear / probabilistic", CFG, build)
import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from src.models.features import add_model_features

MODEL_PATH = "artifacts/final_model.joblib"
COLS_PATH = "artifacts/feature_cols.json"

@lru_cache(maxsize=1)
def _load():
    return joblib.load(MODEL_PATH), json.load(open(COLS_PATH))

def predict_proba(cand_df: pd.DataFrame) -> np.ndarray:
    """B's candidate table (one row per GPS point x candidate) -> P(true arc), same row order."""
    model, cols = _load()
    if len(cand_df) == 0:
        return np.empty(0)
    if not all(c in cand_df.columns for c in cols):
        cand_df = add_model_features(cand_df)
    return model.predict_proba(cand_df[cols].values)[:, 1]
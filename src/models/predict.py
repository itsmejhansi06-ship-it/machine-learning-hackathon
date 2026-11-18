import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "artifacts/final_model.joblib"
COLS_PATH = "artifacts/feature_cols.json"

@lru_cache(maxsize=1)
def _load():
    return joblib.load(MODEL_PATH), json.load(open(COLS_PATH))

def predict_proba(cand_df: pd.DataFrame) -> np.ndarray:
    """
    Score candidate rows: one row per (GPS observation, candidate arc).
    Input : DataFrame with at least the columns in feature_cols.json
            (extra columns such as trip, obs_idx, arc_id, label are ignored).
    Output: 1-D array, P(candidate is the true arc), aligned with cand_df rows.
    """
    model, cols = _load()
    missing = [c for c in cols if c not in cand_df.columns]
    if missing:
        raise ValueError(f"candidate table is missing feature columns: {missing}")
    if len(cand_df) == 0:
        return np.empty(0)
    return model.predict_proba(cand_df[cols].values)[:, 1]
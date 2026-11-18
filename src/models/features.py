import pandas as pd

K_MAX = 10   # candidates per GPS point used by the model and the HMM

# Model inputs. Excluded: projection_epsg (location), raw gps/arc headings (absolute
# compass directions; only their difference is meaningful), cross_track_distance (= distance).
FEATURES = [
    "distance", "distance_rank", "dist_minus_min", "n_cands", "arc_length",
    "gps_speed_mps", "time_delta_s", "heading_diff_deg",
    "projection_fraction", "along_arc_distance",
]


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-observation features on top of B's candidate table. Does not drop rows."""
    df = df.copy()
    g = df.groupby(["trip", "obs_idx"])["distance"]
    df["dist_minus_min"] = df["distance"] - g.transform("min")   # gap to the nearest candidate
    df["n_cands"] = g.transform("size")                          # how ambiguous this point is
    df["heading_diff_deg"] = df["heading_diff_deg"].fillna(90.0) # stationary -> no heading -> neutral
    return df
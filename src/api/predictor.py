import os, logging
import numpy as np
import pandas as pd
from src.data_loader import DATA_ROOT, load_trip
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.preprocessing.candidates import build_candidates
from src.features.feature_engineering import add_candidate_features
from src.models.predict import predict_proba

logger = logging.getLogger("PathSnap")
USE_HMM = os.environ.get("PATHSNAP_HMM", "1") == "1"      # set PATHSNAP_HMM=0 to disable
SPLITS = ["evaluator_test", "student_test", "train"]


def find_trip(trip_id):
    trip_id = str(trip_id).zfill(8)
    for split in SPLITS:
        if (DATA_ROOT / split / trip_id).is_dir():
            return trip_id, split
    raise FileNotFoundError(f"Trip {trip_id} not found")


def list_available_trips():
    for split in ["evaluator_test", "student_test"]:        # evaluator trips first, if present
        d = DATA_ROOT / split
        if d.is_dir():
            return [{"trip_id": f.name, "split": split} for f in sorted(d.iterdir()) if f.is_dir()]
    return []


def predict_trip(trip_id):
    trip_id, split = find_trip(trip_id)
    trip = load_trip(trip_id, split)

    # B: candidates + features (no route used)
    track_p, nodes_p, _ = project_trip(trip["track"], trip["nodes"])
    arc_geoms = build_arc_geometries(nodes_p, trip["arcs"])
    cand = build_candidates(trip, k=10, radius=50.0)
    if cand.empty:
        raise ValueError("No candidate road arcs within 50 m of the GPS track")
    cand = add_candidate_features(cand, track_p, arc_geoms)

    # C: classifier probabilities
    cand["p"] = predict_proba(cand)
    best = cand.loc[cand.groupby("obs_idx")["p"].idxmax()]
    pred = dict(zip(best.obs_idx.astype(int), best.arc_id.astype(int)))
    method = "classifier"

    # D: HMM / Viterbi
    if USE_HMM:
        try:
            from src.hmm.hmm import map_match_trip
            out = map_match_trip(trip, cand_df=cand, max_distance_m=1000)
            pred = dict(zip(out.obs_idx.astype(int), out.predicted_arc_id.astype(int)))
            method = "classifier+hmm"
        except Exception as e:
            logger.warning("HMM failed for %s, using classifier-only: %s", trip_id, e)

    all_obs = trip["track"].obs_idx.astype(int).tolist()
    seq = pd.Series(pred, dtype=float).reindex(all_obs).ffill().bfill().astype(int).to_numpy()

    arcs, nodes = trip["arcs"], trip["nodes"]
    s = seq[np.r_[True, seq[1:] != seq[:-1]]]
    continuity = float((arcs.to_node.values[s[:-1]] == arcs.from_node.values[s[1:]]).mean()) if len(s) > 1 else 1.0

    # Server-side scoring only, after prediction: share of GPS points matched to an arc on the reference route
    top1 = None
    if trip["route"] is not None:
        on_route = set(np.asarray(trip["route"]).tolist())
        top1 = float(np.mean([a in on_route for a in seq]))

    pred_coords = []
    for a in s:
        for n in (arcs.from_node.values[a], arcs.to_node.values[a]):
            c = [float(nodes.lon.values[n]), float(nodes.lat.values[n])]
            if not pred_coords or pred_coords[-1] != c:
                pred_coords.append(c)

    logger.info("trip %s | %s | %d points | on-route acc %s | continuity %.3f",
                trip_id, method, len(seq), None if top1 is None else round(top1, 4), continuity)
    return {"trip_id": trip_id, "split": split, "method": method,
            "pred_coords": pred_coords,
            "gps_coords": trip["track"][["lon", "lat"]].values.tolist(),
            "top1_acc": top1, "continuity": continuity}
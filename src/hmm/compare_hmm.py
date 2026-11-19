
import time
import pandas as pd

from src.api.predictor import find_trip, SPLITS
from src.data_loader import load_trip
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.preprocessing.candidates import build_candidates
from src.features.feature_engineering import add_candidate_features
from src.models.predict import predict_proba
from src.hmm.hmm import map_match_trip


def evaluate(trip, predictions):
    arcs = trip["arcs"]
    route = trip["route"]

    seq = predictions.sort_values("obs_idx")["arc_id"].astype(int).to_numpy()
    unique_seq = seq[pd.Series(seq).shift().ne(seq).to_numpy()]

    continuity = (
        float((arcs.to_node.values[unique_seq[:-1]] ==
               arcs.from_node.values[unique_seq[1:]]).mean())
        if len(unique_seq) > 1 else 1.0
    )

    on_route = (
        float(pd.Series(seq).isin(set(route)).mean())
        if route is not None else None
    )

    return on_route, continuity


def main():
    trips = []
    for split in SPLITS:
        from src.api.predictor import DATA_ROOT
        folder = DATA_ROOT / split
        if folder.exists():
            trips.extend((p.name, split) for p in sorted(folder.iterdir()) if p.is_dir())

    trips = trips[:10]
    results = []

    for trip_id, split in trips:
        print(f"\nProcessing trip {trip_id} ({split})", flush=True)
        trip = load_trip(trip_id, split)

        track_p, nodes_p, _ = project_trip(trip["track"], trip["nodes"])
        arc_geoms = build_arc_geometries(nodes_p, trip["arcs"])
        cand = build_candidates(trip, k=10, radius=50.0)

        if cand.empty:
            print("Skipping: no candidates")
            continue

        cand = add_candidate_features(cand, track_p, arc_geoms)

        start = time.perf_counter()
        cand["p"] = predict_proba(cand)
        best = cand.loc[cand.groupby("obs_idx")["p"].idxmax()]
        classifier = best[["obs_idx", "arc_id"]].copy()
        classifier_time = time.perf_counter() - start

        classifier_acc, classifier_cont = evaluate(trip, classifier)

        start = time.perf_counter()
        hmm_result = map_match_trip(trip, cand_df=cand, max_distance_m=1000)
        hmm_time = time.perf_counter() - start

        hmm = hmm_result.rename(
            columns={"predicted_arc_id": "arc_id"}
        )[["obs_idx", "arc_id"]]

        hmm_acc, hmm_cont = evaluate(trip, hmm)

        results.append({
            "trip_id": trip_id,
            "classifier_on_route_accuracy": classifier_acc,
            "hmm_on_route_accuracy": hmm_acc,
            "classifier_continuity": classifier_cont,
            "hmm_continuity": hmm_cont,
            "classifier_time_sec": classifier_time,
            "hmm_time_sec": hmm_time,
        })

        print(f"Classifier: accuracy={classifier_acc}, continuity={classifier_cont}, time={classifier_time:.2f}s")
        print(f"HMM: accuracy={hmm_acc}, continuity={hmm_cont}, time={hmm_time:.2f}s")

    import os
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/hmm_vs_classifier.csv", index=False)
    print("\nSaved results/hmm_vs_classifier.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

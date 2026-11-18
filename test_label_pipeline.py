
from src.data_loader import load_trip
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.preprocessing.candidates import build_candidates
from src.preprocessing.route_alignment import align_route_to_track
from src.features.labels import attach_candidate_labels


def test_label_pipeline(trip_id, split):
    print("\n" + "=" * 60)
    print(f"Testing candidate labelling: {trip_id} ({split})")
    print("=" * 60)

    trip = load_trip(trip_id, split=split)

    track, nodes, _ = project_trip(
        trip["track"], trip["nodes"]
    )

    arc_geometries = build_arc_geometries(
        nodes, trip["arcs"]
    )

    candidates = build_candidates(
        trip, k=10, radius=50
    )

    alignment = align_route_to_track(
        track, arc_geometries, trip["route"]
    )

    labelled_candidates, summary = attach_candidate_labels(
        candidates, alignment
    )

    print("\nCandidate rows:", len(labelled_candidates))
    print("Positive candidate rows:",
          int(labelled_candidates["label"].sum()))
    print("Negative candidate rows:",
          int((labelled_candidates["label"] == 0).sum()))

    print("\nCandidate miss-rate summary:")
    for key, value in summary.items():
        if key == "candidate_miss_rate":
            print(f"{key}: {value:.4f} ({value * 100:.2f}%)")
        else:
            print(f"{key}: {value}")

    print("\nLabelled candidate sample:")
    print(labelled_candidates.head(10).to_string(index=False))

    return labelled_candidates, alignment, summary


if __name__ == "__main__":
    train_data, train_alignment, train_summary = (
        test_label_pipeline("00000001", "train")
    )

    student_test_data, student_test_alignment, student_test_summary = (
        test_label_pipeline("00000000", "student_test")
    )

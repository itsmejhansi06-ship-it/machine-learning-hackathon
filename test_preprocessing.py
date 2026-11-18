
from src.data_loader import load_trip, get_trip_ids
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import (
    build_arc_geometries,
    build_spatial_index,
)
from src.preprocessing.candidates import build_candidates


def test_trip(trip_id, split):

    print("\n" + "=" * 50)
    print(f"Testing trip: {trip_id} ({split})")
    print("=" * 50)

    trip = load_trip(trip_id, split=split)

    print("GPS observations:", len(trip["track"]))
    print("Road nodes:", len(trip["nodes"]))
    print("Road arcs:", len(trip["arcs"]))
    print("Reference route length:", len(trip["route"]))

    # Coordinate projection
    track, nodes, metadata = project_trip(
        trip["track"],
        trip["nodes"]
    )

    print("\nProjection:", metadata["epsg"])
    print("Projected GPS coordinates:")
    print(track[["x", "y"]].head())

    # Road geometry
    arc_geometries = build_arc_geometries(
        nodes,
        trip["arcs"]
    )

    print("\nConstructed arc geometries:", len(arc_geometries))

    # Spatial index
    tree = build_spatial_index(arc_geometries)

    print("Spatial index:", type(tree).__name__)

    # Candidate generation
    candidates = build_candidates(
        trip,
        k=10,
        radius=50
    )

    print("\nCandidate rows:", len(candidates))

    if not candidates.empty:

        print(
            "Observations with candidates:",
            candidates["obs_idx"].nunique()
        )

        print(
            "Average candidates per matched observation:",
            round(
                len(candidates)
                / candidates["obs_idx"].nunique(),
                2
            )
        )

        print("\nFirst candidate rows:")
        print(candidates.head(10))

    else:
        print("WARNING: No candidates generated!")

    # Check observation coverage
    total_observations = len(trip["track"])
    covered = (
        candidates["obs_idx"].nunique()
        if not candidates.empty else 0
    )

    no_candidate = total_observations - covered

    print("\nObservations without candidates:", no_candidate)
    print(
        "Candidate availability:",
        round(covered / total_observations * 100, 2),
        "%"
    )

    return candidates


if __name__ == "__main__":

    print("Training trips:", len(get_trip_ids("train")))
    print(
        "Student-test trips:",
        len(get_trip_ids("student_test"))
    )

    # First training trip
    train_candidates = test_trip(
        "00000001",
        "train"
    )

    # First student-test trip
    test_candidates = test_trip(
        "00000000",
        "student_test"
    )

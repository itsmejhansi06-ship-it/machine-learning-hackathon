
from src.data_loader import load_trip
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.preprocessing.route_alignment import align_route_to_track


def test_alignment(trip_id, split):
    print("\n" + "=" * 55)
    print(f"Testing route alignment: {trip_id} ({split})")
    print("=" * 55)

    trip = load_trip(trip_id, split=split)

    track, nodes, projection = project_trip(
        trip["track"], trip["nodes"]
    )

    arc_geometries = build_arc_geometries(
        nodes, trip["arcs"]
    )

    alignment = align_route_to_track(
        track,
        arc_geometries,
        trip["route"]
    )

    print("GPS observations:", len(track))
    print("Reference route arcs:", len(trip["route"]))
    print("Aligned observations:", len(alignment))

    print("\nAlignment distance statistics (metres):")
    print(alignment["alignment_distance"].describe())

    print("\nFirst 10 aligned observations:")
    print(alignment.head(10).to_string(index=False))

    print("\nLast 10 aligned observations:")
    print(alignment.tail(10).to_string(index=False))

    print("\nUnique reference arcs assigned:",
          alignment["reference_arc_id"].nunique())

    return alignment


if __name__ == "__main__":
    train_alignment = test_alignment("00000001", "train")
    test_alignment_result = test_alignment(
        "00000000", "student_test"
    )

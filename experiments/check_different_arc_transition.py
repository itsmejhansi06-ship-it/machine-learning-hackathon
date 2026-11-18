
import sys
import math
from pathlib import Path

from shapely.geometry import Point

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.preprocessing.candidates import build_candidates
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.hmm.road_graph import (
    load_road_graph,
    complete_network_distance
)


def main():

    print("\nDIFFERENT ARC TRANSITION TEST")
    print("-" * 45)

    # Step 1: Load trip
    trip = load_trip("00000001", split="train")

    print("Trip loaded:", trip["trip_id"])

    # Step 2: Generate candidates
    candidates = build_candidates(trip)

    # Step 3: Project coordinates
    track_projected, nodes_projected, projection = project_trip(
        trip["track"],
        trip["nodes"]
    )

    # Step 4: Build road geometries
    arc_geometries = build_arc_geometries(
        nodes_projected,
        trip["arcs"]
    )

    geometry_lookup = {
        int(row.arc_id): row.geometry
        for row in arc_geometries.itertuples(index=False)
    }

    # Step 5: Build graph
    graph, arc_lookup = load_road_graph(trip)

    # Step 6: Select candidates for observations 0 and 1
    first_candidates = candidates[
        candidates["obs_idx"] == 0
    ]

    second_candidates = candidates[
        candidates["obs_idx"] == 1
    ]

    # Find a pair of different candidate arcs
    selected_pair = None

    for _, previous in first_candidates.iterrows():

        for _, current in second_candidates.iterrows():

            previous_arc = int(previous["arc_id"])
            current_arc = int(current["arc_id"])

            if previous_arc != current_arc:
                selected_pair = (previous, current)
                break

        if selected_pair is not None:
            break

    if selected_pair is None:
        raise ValueError("No different-arc candidate pair found.")

    previous, current = selected_pair

    previous_arc = int(previous["arc_id"])
    current_arc = int(current["arc_id"])

    print("\nSelected arcs:")
    print("Previous arc:", previous_arc)
    print("Current arc:", current_arc)

    # Step 7: Get projected GPS coordinates
    previous_gps = track_projected[
        track_projected["obs_idx"] == 0
    ].iloc[0]

    current_gps = track_projected[
        track_projected["obs_idx"] == 1
    ].iloc[0]

    previous_x = float(previous_gps["x"])
    previous_y = float(previous_gps["y"])

    current_x = float(current_gps["x"])
    current_y = float(current_gps["y"])

    previous_geometry = geometry_lookup[previous_arc]
    current_geometry = geometry_lookup[current_arc]

    # Step 8: Calculate component distances
    previous_position = previous_geometry.project(
        Point(previous_x, previous_y)
    )

    current_position = current_geometry.project(
        Point(current_x, current_y)
    )

    remaining_previous = (
        previous_geometry.length - previous_position
    )

    initial_current = current_position

    print("\nDistance components:")
    print("Remaining previous arc:", remaining_previous)
    print("Initial current arc:", initial_current)

    # Step 9: Calculate connecting distance
    from src.hmm.road_graph import arc_network_distance

    connecting_distance = arc_network_distance(
        graph,
        arc_lookup,
        previous_arc,
        current_arc,
        max_distance_m=5000
    )

    print("Connecting network distance:", connecting_distance)

    # Step 10: Calculate complete network distance
    distance = complete_network_distance(
        graph=graph,
        arc_lookup=arc_lookup,
        previous_arc=previous_arc,
        current_arc=current_arc,
        previous_geometry=previous_geometry,
        current_geometry=current_geometry,
        previous_x=previous_x,
        previous_y=previous_y,
        current_x=current_x,
        current_y=current_y,
        max_distance_m=5000
    )

    print("\nRESULT:")
    print("Complete network distance:", distance)

    if math.isfinite(distance) and distance >= 0:
        print("\nSUCCESS: Different-arc distance is valid!")
    else:
        print("\nNo valid directed path found for this candidate pair.")


if __name__ == "__main__":
    main()

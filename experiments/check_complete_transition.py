
import sys
import math
import inspect
from pathlib import Path

from shapely.geometry import Point

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import src.hmm.road_graph as road_graph_module

from src.data_loader import load_trip
from src.preprocessing.candidates import build_candidates
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.hmm.road_graph import (
    load_road_graph,
    complete_network_distance
)


def main():

    print("\nCOMPLETE NETWORK TRANSITION TEST")
    print("-" * 45)

    # Step 1: Load a training trip
    trip = load_trip("00000001", split="train")

    print("Trip loaded:", trip["trip_id"])

    # Step 2: Generate candidate arcs
    candidates = build_candidates(trip)

    print("Candidate rows:", len(candidates))

    # Step 3: Project GPS observations and road nodes
    track_projected, nodes_projected, projection = project_trip(
        trip["track"],
        trip["nodes"]
    )

    print("Projection EPSG:", projection["epsg"])

    # Step 4: Build road arc geometries
    arc_geometries = build_arc_geometries(
        nodes_projected,
        trip["arcs"]
    )

    geometry_lookup = {
        int(row.arc_id): row.geometry
        for row in arc_geometries.itertuples(index=False)
    }

    print("Road arc geometries:", len(geometry_lookup))

    # Step 5: Build directed road graph
    graph, arc_lookup = load_road_graph(trip)

    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    # Step 6: Select candidates for observations 0 and 1
    first_candidates = candidates[
        candidates["obs_idx"] == 0
    ]

    second_candidates = candidates[
        candidates["obs_idx"] == 1
    ]

    if first_candidates.empty or second_candidates.empty:
        raise ValueError(
            "No candidates found for the first two observations."
        )

    previous = first_candidates.iloc[0]
    current = second_candidates.iloc[0]

    previous_arc = int(previous["arc_id"])
    current_arc = int(current["arc_id"])

    print("\nSelected candidates:")
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

    # Step 8: Get corresponding road geometries
    previous_geometry = geometry_lookup[previous_arc]
    current_geometry = geometry_lookup[current_arc]

    # Step 9: Inspect projected positions
    previous_position = previous_geometry.project(
        Point(previous_x, previous_y)
    )

    current_position = current_geometry.project(
        Point(current_x, current_y)
    )

    print("\nDEBUG INFORMATION:")
    print("Previous arc:", previous_arc)
    print("Current arc:", current_arc)
    print("Arc length:", previous_geometry.length)
    print("Previous position:", previous_position)
    print("Current position:", current_position)
    print(
        "Position difference:",
        current_position - previous_position
    )

    print("\nFirst 5 candidates for observation 0:")
    print(
        first_candidates[
            ["arc_id", "distance", "distance_rank"]
        ].head()
    )

    print("\nFirst 5 candidates for observation 1:")
    print(
        second_candidates[
            ["arc_id", "distance", "distance_rank"]
        ].head()
    )

    # Step 10: Function diagnostics
    print("\nFUNCTION DIAGNOSTICS:")
    print("Previous arc exists:", previous_arc in arc_lookup)
    print("Current arc exists:", current_arc in arc_lookup)
    print("Same arc:", previous_arc == current_arc)
    print("Previous arc type:", type(previous_arc))
    print("Arc lookup key example:", list(arc_lookup.keys())[:5])

    expected_distance = current_position - previous_position

    print("Expected same-arc distance:", expected_distance)
    print(
        "Distance within cap:",
        0 <= expected_distance <= 5000
    )

    # Step 11: Inspect the actual imported function
    print("\nIMPORTED FUNCTION CHECK:")
    print("Module location:", road_graph_module.__file__)

    print("\nActual imported function:")
    print(
        inspect.getsource(
            road_graph_module.complete_network_distance
        )
    )

    # Step 12: Calculate distance directly
    print("\nDIRECT SAME-ARC CALCULATION:")

    direct_distance = (
        current_geometry.project(
            Point(current_x, current_y)
        )
        -
        previous_geometry.project(
            Point(previous_x, previous_y)
        )
    )

    print("Direct result:", direct_distance)

    # Step 13: Call the actual imported function
    print("\nCALLING COMPLETE NETWORK DISTANCE:")

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

    # Step 14: Calculate straight-line GPS distance
    gps_distance = math.hypot(
        current_x - previous_x,
        current_y - previous_y
    )

    print("\nDISTANCE RESULTS:")
    print(f"GPS straight-line distance: {gps_distance:.2f} m")
    print(f"Direct same-arc distance: {direct_distance:.2f} m")
    print(f"Complete network distance: {distance:.2f} m")

    # Step 15: Validate
    if math.isfinite(distance) and distance >= 0:
        print("\nSUCCESS: Complete network distance is valid!")
    else:
        print("\nFAILED: Complete network distance is invalid.")


if __name__ == "__main__":
    main()

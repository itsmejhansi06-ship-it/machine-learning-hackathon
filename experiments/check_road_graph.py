
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.hmm.road_graph import (
    load_road_graph,
    dijkstra_distance,
    arc_network_distance
)

TRIP_ID = "00000001"

TRIP_DIR = Path(
     "data/raw/Student folder/Student folder/train/00000001"
)

# Build road graph
graph, arc_lookup = load_road_graph(
    TRIP_DIR,
    TRIP_ID
)

print("ROAD GRAPH")
print("Nodes:", graph.number_of_nodes())
print("Arcs:", graph.number_of_edges())

# Read reference route for structural testing
route = np.loadtxt(
    TRIP_DIR / f"{TRIP_ID}.route",
    dtype=int
)

# Check consecutive reference arcs
print("\nROUTE CONNECTIVITY")

for previous_arc, current_arc in zip(
    route[:5], route[1:6]
):
    distance = arc_network_distance(
        graph,
        arc_lookup,
        int(previous_arc),
        int(current_arc)
    )

    print(
        f"Arc {previous_arc} -> {current_arc}: "
        f"connecting distance = {distance:.2f} m"
    )

# Test Dijkstra

# Test Dijkstra across an intermediate road arc

first_arc = int(route[0])
third_arc = int(route[2])

source = arc_lookup[first_arc]["to_node"]
target = arc_lookup[third_arc]["from_node"]

distance = dijkstra_distance(
    graph,
    source,
    target,
    max_distance_m=5000
)

expected_distance = arc_lookup[int(route[1])]["length_m"]

print("\nDIJKSTRA TEST")
print("Source:", source)
print("Target:", target)
print("Calculated distance:", distance)
print("Expected distance:", expected_distance)

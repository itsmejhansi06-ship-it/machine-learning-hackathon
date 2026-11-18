
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.hmm.road_graph import (
    load_road_graph,
    dijkstra_distance,
    arc_network_distance
)

TRIP_ID = "00000001"

# Load trip using the shared data loader
trip = load_trip(TRIP_ID, split="train")

# Build directed road graph
graph, arc_lookup = load_road_graph(trip)

print("ROAD GRAPH")
print("Nodes:", graph.number_of_nodes())
print("Arcs:", graph.number_of_edges())

# Get reference route
route = trip["route"]

print("\nROUTE CONNECTIVITY")

for previous_arc, current_arc in zip(route[:5], route[1:6]):
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

# Dijkstra test across an intermediate road arc
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

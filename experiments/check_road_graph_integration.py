
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.hmm.road_graph import load_road_graph


TRIP_ID = "00000001"

# Load trip using Person B's data loader
trip = load_trip(TRIP_ID, split="train")

# Build graph directly from the trip dictionary
graph, arc_lookup = load_road_graph(trip)

print("ROAD GRAPH INTEGRATION TEST")
print("---------------------------")

print("Trip ID:", trip["trip_id"])
print("Nodes loaded:", len(trip["nodes"]))
print("Arcs loaded:", len(trip["arcs"]))

print("Graph nodes:", graph.number_of_nodes())
print("Graph edges:", graph.number_of_edges())

print("Arc lookup entries:", len(arc_lookup))

assert graph.number_of_nodes() == len(trip["nodes"])
assert graph.number_of_edges() == len(trip["arcs"])
assert len(arc_lookup) == len(trip["arcs"])

print("\nAll integration checks passed!")

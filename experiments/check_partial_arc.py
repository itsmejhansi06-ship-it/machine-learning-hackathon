
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from shapely.geometry import LineString
from src.hmm.road_graph import partial_arc_distance


# Example road arc from (0, 0) to (100, 0)
arc = LineString([
    (0, 0),
    (100, 0)
])

# GPS point located 30 metres from the start
point_x = 30
point_y = 0

initial = partial_arc_distance(
    arc, point_x, point_y, "initial"
)

remaining = partial_arc_distance(
    arc, point_x, point_y, "remaining"
)

print("PARTIAL ARC DISTANCE TEST")
print("-------------------------")
print("Initial distance:", initial)
print("Remaining distance:", remaining)
print("Total arc length:", arc.length)

assert initial == 30
assert remaining == 70

print("\nAll tests passed!")

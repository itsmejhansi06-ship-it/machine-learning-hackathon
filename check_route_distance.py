
import numpy as np
from shapely.geometry import Point

from src.data_loader import load_trip
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries

trip = load_trip("00000009", split="student_test")

track, nodes, projection = project_trip(
    trip["track"],
    trip["nodes"]
)

arc_geometries = build_arc_geometries(
    nodes,
    trip["arcs"]
)

arc_lookup = arc_geometries.set_index("arc_id")

# Construct the complete reference route geometry list
route_geometries = [
    arc_lookup.loc[arc_id, "geometry"]
    for arc_id in trip["route"]
]

print("GPS distance to nearest reference-route arc")
print("-" * 65)

records = []

for observation in track.itertuples(index=False):
    point = Point(observation.x, observation.y)

    distances = [
        point.distance(geometry)
        for geometry in route_geometries
    ]

    nearest_position = int(np.argmin(distances))
    nearest_distance = distances[nearest_position]

    records.append({
        "obs_idx": observation.obs_idx,
        "nearest_route_position": nearest_position,
        "nearest_route_arc": trip["route"][nearest_position],
        "nearest_route_distance": nearest_distance
    })

import pandas as pd

result = pd.DataFrame(records)

print("\nFirst 20 observations:")
print(result.head(20).to_string(index=False))

print("\nDistance statistics:")
print(result["nearest_route_distance"].describe())

print("\nObservations more than 50 metres from the entire reference route:")
print(
    (result["nearest_route_distance"] > 50).sum()
)

print("\nObservations more than 100 metres from the entire reference route:")
print(
    (result["nearest_route_distance"] > 100).sum()
)

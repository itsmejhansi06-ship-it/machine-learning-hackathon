
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.data_loader import load_trip
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.preprocessing.route_alignment import align_route_to_track


trip = load_trip("00000009", split="student_test")

track, nodes, projection = project_trip(
    trip["track"],
    trip["nodes"]
)

arc_geometries = build_arc_geometries(nodes, trip["arcs"])

# Existing DTW alignment
dtw_alignment = align_route_to_track(
    track,
    arc_geometries,
    trip["route"]
)

arc_lookup = arc_geometries.set_index("arc_id")

route_geometries = [
    arc_lookup.loc[arc_id, "geometry"]
    for arc_id in trip["route"]
]

# Find nearest route position for each GPS observation
nearest_records = []

for obs in track.itertuples(index=False):
    point = Point(obs.x, obs.y)

    distances = [
        point.distance(geometry)
        for geometry in route_geometries
    ]

    nearest_pos = int(np.argmin(distances))

    nearest_records.append({
        "obs_idx": obs.obs_idx,
        "nearest_route_position": nearest_pos,
        "nearest_route_distance": distances[nearest_pos]
    })

nearest_df = pd.DataFrame(nearest_records)

comparison = dtw_alignment.merge(
    nearest_df,
    on="obs_idx"
)

comparison["position_difference"] = (
    comparison["route_position"]
    - comparison["nearest_route_position"]
)

print("\nAlignment comparison:")
print(
    comparison[
        [
            "obs_idx",
            "route_position",
            "nearest_route_position",
            "position_difference",
            "alignment_distance",
            "nearest_route_distance"
        ]
    ].head(30).to_string(index=False)
)

print("\nPosition difference statistics:")
print(comparison["position_difference"].describe())

print("\nObservations where DTW is more than 2 route positions ahead:")
print(
    (comparison["position_difference"] > 2).sum()
)

print("\nObservations where DTW is more than 2 route positions behind:")
print(
    (comparison["position_difference"] < -2).sum()
)


import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import (
    build_arc_geometries,
    build_spatial_index,
)


DEFAULT_K = 10
DEFAULT_RADIUS = 50.0


def build_candidates(
    trip,
    k=DEFAULT_K,
    radius=DEFAULT_RADIUS
):
    """
    Generate candidate road arcs for every GPS observation.

    Parameters
    ----------
    trip : dict
        Output of load_trip().
    k : int
        Maximum number of candidate arcs per observation.
    radius : float
        Maximum point-to-arc distance in metres.

    Returns
    -------
    candidates : pandas.DataFrame
        One row per (observation, candidate arc).
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    if radius <= 0:
        raise ValueError("radius must be positive.")

    track, nodes, projection = project_trip(
        trip["track"],
        trip["nodes"]
    )

    arc_geometries = build_arc_geometries(
        nodes,
        trip["arcs"]
    )

    tree = build_spatial_index(arc_geometries)

    geometries = arc_geometries["geometry"].tolist()

    records = []

    for observation in track.itertuples(index=False):

        point = Point(observation.x, observation.y)

        # Query only arcs whose bounding boxes intersect
        # the radius buffer.
        possible_indices = tree.query(
            point.buffer(radius)
        )

        nearby = []

        for idx in possible_indices:

            idx = int(idx)
            geometry = geometries[idx]

            distance = point.distance(geometry)

            if distance <= radius:
                nearby.append((idx, distance))

        # Sort by perpendicular distance.
        nearby.sort(key=lambda item: item[1])

        # Keep only top-k candidates.
        nearby = nearby[:k]

        for rank, (idx, distance) in enumerate(nearby, start=1):

            arc = arc_geometries.iloc[idx]

            records.append({
                "trip": trip["trip_id"],
                "obs_idx": observation.obs_idx,
                "t": observation.t,
                "arc_id": arc.arc_id,
                "distance": distance,
                "distance_rank": rank,
                "arc_length": arc.arc_length,
            })

    candidates = pd.DataFrame(records)

    if not candidates.empty:
        candidates["trip"] = candidates["trip"].astype(str)
        candidates["obs_idx"] = candidates["obs_idx"].astype(int)
        candidates["arc_id"] = candidates["arc_id"].astype(int)

    return candidates

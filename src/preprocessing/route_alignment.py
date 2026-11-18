
import numpy as np
import pandas as pd
from shapely.geometry import Point


def align_route_to_track(track, arc_geometries, route):
    """
    Align GPS observations to the ordered reference route using
    Dynamic Time Warping (DTW).

    Returns one reference arc ID for each GPS observation,
    along with the distance to that arc.
    """

    if len(track) == 0:
        raise ValueError("GPS track is empty.")

    if len(route) == 0:
        raise ValueError("Reference route is empty.")

    # Map each arc ID to its geometry.
    arc_lookup = arc_geometries.set_index("arc_id")["geometry"]

    # Preserve the route order, including repeated arc IDs.
    try:
        route_geometries = [
            arc_lookup.loc[int(arc_id)] for arc_id in route
        ]
    except KeyError as exc:
        raise ValueError(
            f"Route references an arc without geometry: {exc}"
        ) from exc

    gps_points = [
        Point(x, y)
        for x, y in zip(track["x"], track["y"])
    ]

    n = len(gps_points)
    m = len(route_geometries)

    # Local cost: distance from each GPS observation to each route arc.
    costs = np.empty((n, m), dtype=np.float64)

    for i, point in enumerate(gps_points):
        for j, geometry in enumerate(route_geometries):
            costs[i, j] = point.distance(geometry)

    # DTW cumulative cost matrix.
    dp = np.full((n, m), np.inf, dtype=np.float64)
    predecessor = np.full((n, m), -1, dtype=np.int8)

    dp[0, 0] = costs[0, 0]

    # First column: multiple GPS observations can map to one route arc.
    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + costs[i, 0]
        predecessor[i, 0] = 0

    # First row: route arcs may be skipped before the first observation.
    for j in range(1, m):
        dp[0, j] = dp[0, j - 1] + costs[0, j]
        predecessor[0, j] = 1

    # 0 = previous GPS observation, same route position
    # 1 = same GPS observation, previous route position
    # 2 = previous GPS observation, previous route position
    for i in range(1, n):
        for j in range(1, m):
            previous = (
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

            best = int(np.argmin(previous))
            dp[i, j] = costs[i, j] + previous[best]
            predecessor[i, j] = best

    # Backtrack through the optimal alignment.
    i, j = n - 1, m - 1
    aligned_route_positions = [[] for _ in range(n)]

    while True:
        aligned_route_positions[i].append(j)

        if i == 0 and j == 0:
            break

        move = predecessor[i, j]

        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        elif move == 2:
            i -= 1
            j -= 1
        else:
            raise RuntimeError("Invalid DTW predecessor.")

    # A DTW path can visit multiple route positions for one GPS point.
    # Choose the visited route arc closest to that GPS point.
    records = []

    for obs_idx, positions in enumerate(aligned_route_positions):
        if not positions:
            raise RuntimeError(
                f"Observation {obs_idx} was not aligned."
            )

        best_position = min(
            positions,
            key=lambda pos: costs[obs_idx, pos]
        )

        records.append({
            "obs_idx": int(track.iloc[obs_idx]["obs_idx"]),
            "reference_arc_id": int(route[best_position]),
            "alignment_distance": float(
                costs[obs_idx, best_position]
            ),
            "route_position": int(best_position),
        })

    return pd.DataFrame(records)

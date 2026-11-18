
import numpy as np
import pandas as pd
from shapely.geometry import Point


def _bearing_deg(dx, dy):
    """Bearing clockwise from north, in degrees."""
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def _gps_motion_features(track):
    """
    Calculate GPS speed and heading for every observation.

    Uses the next observation, except for the final observation,
    which uses the previous one.
    """

    track = track.sort_values("obs_idx").copy()

    x = track["x"].to_numpy(dtype=float)
    y = track["y"].to_numpy(dtype=float)
    t = track["t"].to_numpy(dtype=float)

    n = len(track)
    speeds = np.full(n, np.nan)
    headings = np.full(n, np.nan)
    time_deltas = np.full(n, np.nan)

    for i in range(n):
        if n < 2:
            continue

        if i < n - 1:
            j = i + 1
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dt = t[j] - t[i]
        else:
            j = i - 1
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dt = t[i] - t[j]

        if dt == 0:
            continue

        speeds[i] = np.hypot(dx, dy) / abs(dt)
        time_deltas[i] = abs(dt)

        if np.hypot(dx, dy) > 0.01:
            headings[i] = _bearing_deg(dx, dy)

    return pd.DataFrame({
        "obs_idx": track["obs_idx"].to_numpy(),
        "gps_speed_mps": speeds,
        "gps_heading_deg": headings,
        "time_delta_s": time_deltas,
    })


def add_candidate_features(candidates, track, arc_geometries):
    """
    Add movement and road-geometry features to candidate rows.

    track must contain obs_idx, t, x and y.
    arc_geometries must contain arc_id, geometry and arc_length.
    """

    if candidates.empty:
        return candidates.copy()

    required_track = {"obs_idx", "t", "x", "y"}
    required_arcs = {"arc_id", "geometry", "arc_length"}

    if not required_track.issubset(track.columns):
        raise ValueError("Projected track is missing required columns.")

    if not required_arcs.issubset(arc_geometries.columns):
        raise ValueError("Arc geometries are missing required columns.")

    if arc_geometries["arc_id"].duplicated().any():
        raise ValueError("Arc IDs must be unique.")

    motion = _gps_motion_features(track)

    result = candidates.merge(
        track[["obs_idx", "x", "y"]],
        on="obs_idx",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        motion,
        on="obs_idx",
        how="left",
        validate="many_to_one",
    )

    geometry_lookup = arc_geometries.set_index("arc_id")["geometry"]

    arc_headings = {}
    projection_fractions = []
    along_arc_distances = []
    cross_track_distances = []

    for row in result.itertuples(index=False):
        geometry = geometry_lookup.loc[row.arc_id]

        if geometry.length <= 0:
            arc_heading = np.nan
            fraction = np.nan
            along_distance = np.nan
            cross_distance = np.nan
        else:
            coords = list(geometry.coords)
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]

            arc_heading = _bearing_deg(dx, dy)

            point = Point(row.x, row.y)
            along_distance = geometry.project(point)
            fraction = along_distance / geometry.length
            cross_distance = point.distance(geometry)

        arc_headings[row.arc_id] = arc_heading
        projection_fractions.append(fraction)
        along_arc_distances.append(along_distance)
        cross_track_distances.append(cross_distance)

    result["arc_heading_deg"] = result["arc_id"].map(arc_headings)
    result["projection_fraction"] = projection_fractions
    result["along_arc_distance"] = along_arc_distances
    result["cross_track_distance"] = cross_track_distances

    heading_difference = (
        result["gps_heading_deg"] - result["arc_heading_deg"]
    ).abs()

    result["heading_diff_deg"] = np.minimum(
        heading_difference,
        360 - heading_difference,
    )

    result = result.drop(columns=["x", "y"])

    return result

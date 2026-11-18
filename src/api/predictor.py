
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

DATA_ROOT = Path(__file__).resolve().parents[2] / "PathSnap" / "Student folder"


def get_utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def find_trip(trip_id):
    trip_id = str(trip_id).zfill(8)

    for split in ["train", "student_test"]:
        folder = DATA_ROOT / split / trip_id

        if folder.exists():
            return folder, split

    raise FileNotFoundError(
        f"Trip {trip_id} not found in available dataset"
    )


def list_available_trips():
    trips = []

    for split in ["train", "student_test"]:
        split_dir = DATA_ROOT / split

        if not split_dir.exists():
            continue

        for folder in sorted(split_dir.iterdir()):
            if folder.is_dir() and folder.name.isdigit():
                trips.append({
                    "trip_id": folder.name,
                    "split": split
                })

    return trips


def predict_trip(trip_id):
    folder, split = find_trip(trip_id)
    trip_id = folder.name

    track_path = folder / f"{trip_id}.track"
    nodes_path = folder / f"{trip_id}.nodes"
    arcs_path = folder / f"{trip_id}.arcs"

    track = pd.read_csv(track_path, sep=r"\s+", header=None)
    nodes = pd.read_csv(nodes_path, sep=r"\s+", header=None)
    arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None)

    gps = track.iloc[:, :2].to_numpy(dtype=float)
    node_coords = nodes.iloc[:, :2].to_numpy(dtype=float)
    arc_indices = arcs.iloc[:, :2].to_numpy(dtype=int)

    if len(gps) == 0 or len(arc_indices) == 0:
        raise ValueError("Trip has no GPS observations or road arcs")

    # Project coordinates into the appropriate UTM zone.
    center_lon = float(np.mean(gps[:, 0]))
    center_lat = float(np.mean(gps[:, 1]))
    epsg = get_utm_epsg(center_lon, center_lat)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )

    gps_x, gps_y = transformer.transform(
        gps[:, 0], gps[:, 1]
    )
    node_x, node_y = transformer.transform(
        node_coords[:, 0], node_coords[:, 1]
    )

    projected_nodes = np.column_stack([node_x, node_y])
    projected_gps = np.column_stack([gps_x, gps_y])

    # Construct road segments.
    valid = (
        (arc_indices[:, 0] >= 0) &
        (arc_indices[:, 1] >= 0) &
        (arc_indices[:, 0] < len(projected_nodes)) &
        (arc_indices[:, 1] < len(projected_nodes))
    )

    valid_arcs = arc_indices[valid]

    if len(valid_arcs) == 0:
        raise ValueError("No valid road arcs found")

    lines = [
        LineString([
            projected_nodes[u],
            projected_nodes[v]
        ])
        for u, v in valid_arcs
    ]

    tree = STRtree(lines)

    predicted_arc_indices = []

    # Temporary nearest-arc prediction.
    for x, y in projected_gps:
        nearest_index = int(tree.nearest(Point(x, y)))
        predicted_arc_indices.append(nearest_index)

    # Calculate route continuity.
    connected = 0
    total_transitions = max(
        0, len(predicted_arc_indices) - 1
    )

    for i in range(total_transitions):
        current_arc = valid_arcs[predicted_arc_indices[i]]
        next_arc = valid_arcs[predicted_arc_indices[i + 1]]

        if current_arc[1] == next_arc[0]:
            connected += 1

    continuity = (
        connected / total_transitions
        if total_transitions else 1.0
    )

    # Build predicted route coordinates.
    pred_coords = []

    for index in predicted_arc_indices:
        u, v = valid_arcs[index]

        for node_id in [u, v]:
            coord = [
                float(node_coords[node_id, 0]),
                float(node_coords[node_id, 1])
            ]

            if not pred_coords or coord != pred_coords[-1]:
                pred_coords.append(coord)

    gps_coords = [
        [float(lon), float(lat)]
        for lon, lat in gps
    ]

    return {
        "trip_id": trip_id,
        "pred_coords": pred_coords,
        "gps_coords": gps_coords,
        "top1_acc": None,
        "continuity": float(continuity),
        "split": split
    }


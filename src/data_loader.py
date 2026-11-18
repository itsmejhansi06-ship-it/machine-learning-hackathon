
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# Raw file column names
TRACK_COLUMNS = ["lon", "lat", "t"]
NODE_COLUMNS = ["lon", "lat"]
ARC_COLUMNS = ["from_node", "to_node"]


def get_trip_ids(split="train", data_root=DATA_ROOT):
    """
    Return sorted trip IDs available in a partition.

    split: 'train' or 'student_test'
    """
    split_dir = Path(data_root) / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Partition directory not found: {split_dir}"
        )

    return sorted(
        folder.name
        for folder in split_dir.iterdir()
        if folder.is_dir()
    )


def _trip_file(trip_id, extension, split, data_root):
    """Construct and validate the path to a trip file."""

    path = (
        Path(data_root)
        / split
        / str(trip_id)
        / f"{trip_id}.{extension}"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    return path


def load_track(trip_id, split="train", data_root=DATA_ROOT):
    """Load GPS observations."""

    path = _trip_file(trip_id, "track", split, data_root)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=TRACK_COLUMNS,
        dtype={
            "lon": np.float64,
            "lat": np.float64,
            "t": np.float64,
        },
    )

    df.insert(0, "obs_idx", np.arange(len(df)))

    return df


def load_nodes(trip_id, split="train", data_root=DATA_ROOT):
    """Load road-network nodes. Row index is node_id."""

    path = _trip_file(trip_id, "nodes", split, data_root)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=NODE_COLUMNS,
        dtype=np.float64,
    )

    df.insert(0, "node_id", np.arange(len(df)))

    return df


def load_arcs(trip_id, split="train", data_root=DATA_ROOT):
    """Load directed road arcs. Row index is arc_id."""

    path = _trip_file(trip_id, "arcs", split, data_root)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ARC_COLUMNS,
        dtype=np.int64,
    )

    df.insert(0, "arc_id", np.arange(len(df)))

    return df


def load_route(trip_id, split="train", data_root=DATA_ROOT):
    """Load the ordered reference route."""

    path = _trip_file(trip_id, "route", split, data_root)

    route = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["arc_id"],
        dtype=np.int64,
    )

    return route["arc_id"].to_numpy()


def load_trip(trip_id, split="train", data_root=DATA_ROOT):
    """
    Load all raw data belonging to one trip.

    Returns a dictionary containing:
    track, nodes, arcs and route.
    """

    track = load_track(trip_id, split, data_root)
    nodes = load_nodes(trip_id, split, data_root)
    arcs = load_arcs(trip_id, split, data_root)
    route = load_route(trip_id, split, data_root)

    # Validate road-network references
    if len(arcs):
        if arcs[["from_node", "to_node"]].min().min() < 0:
            raise ValueError("Negative node ID found.")

        if arcs[["from_node", "to_node"]].max().max() >= len(nodes):
            raise ValueError("Arc references a nonexistent node.")

    if len(route):
        if route.min() < 0 or route.max() >= len(arcs):
            raise ValueError("Route references a nonexistent arc.")

    return {
        "trip_id": str(trip_id),
        "split": split,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }

#!/usr/bin/env python3
"""
PathSnap Person A - Step 1 EDA
Reads the extracted "Student folder" directory and creates:
  results/eda_inventory.csv
  results/route_connectivity.csv
  results/eda_summary.txt
  results/trip_<TRIP_ID>_eda_map.html

Install:
  python3 -m pip install pandas numpy networkx folium pyproj

Run from the project directory:
  python3 person_a_eda.py --data "Student folder" --trip 00000034
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import folium
from folium import FeatureGroup, LayerControl
from pyproj import Transformer


def read_table(path: Path, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype)


def get_utm_epsg(lon: float, lat: float) -> int:
    # UTM zones are 1..60. EPSG 326xx is northern hemisphere;
    # EPSG 327xx is southern hemisphere.
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def analyse_trip(split_dir: Path, trip_id: str) -> tuple[dict, dict]:
    folder = split_dir / trip_id
    base = folder / trip_id
    track = read_table(Path(str(base) + ".track"))
    nodes = read_table(Path(str(base) + ".nodes"))
    arcs = read_table(Path(str(base) + ".arcs"), dtype=int)
    route = read_table(Path(str(base) + ".route"), dtype=int)

    lon = track.iloc[:, 0].to_numpy(dtype=float)
    lat = track.iloc[:, 1].to_numpy(dtype=float)
    ts = track.iloc[:, 2].to_numpy(dtype=float)
    dt = np.diff(ts)

    arc_arr = arcs.to_numpy(dtype=int)
    route_ids = route.iloc[:, 0].to_numpy(dtype=int)
    valid_route_mask = (route_ids >= 0) & (route_ids < len(arc_arr))
    valid_route_ids = route_ids[valid_route_mask]
    route_edges = arc_arr[valid_route_ids] if len(valid_route_ids) else np.empty((0, 2), dtype=int)
    connected = int(np.sum(route_edges[:-1, 1] == route_edges[1:, 0])) if len(route_edges) > 1 else 0
    route_pairs = max(0, len(route_edges) - 1)
    disconnected = route_pairs - connected

    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(arc_arr.tolist())
    components = nx.number_connected_components(graph)
    isolated = len(list(nx.isolates(graph)))

    center_lon = float(np.mean(lon))
    center_lat = float(np.mean(lat))
    epsg = get_utm_epsg(center_lon, center_lat)

    inventory = {
        "split": split_dir.name,
        "trip": trip_id,
        "gps_observations": len(track),
        "node_count": len(nodes),
        "arc_count": len(arcs),
        "route_arc_count": len(route),
        "track_nan_cells": int(track.isna().sum().sum()),
        "duplicate_gps_rows": int(track.duplicated().sum()),
        "timestamp_non_increasing_steps": int(np.sum(dt <= 0)),
        "median_delta_t_seconds": float(np.median(dt)) if len(dt) else np.nan,
        "pct_delta_t_exactly_1s": float(np.mean(np.isclose(dt, 1.0)) * 100) if len(dt) else np.nan,
        "max_delta_t_seconds": float(np.max(dt)) if len(dt) else np.nan,
        "longitude_min": float(np.min(lon)),
        "longitude_max": float(np.max(lon)),
        "latitude_min": float(np.min(lat)),
        "latitude_max": float(np.max(lat)),
        "route_invalid_arc_ids": int(np.sum(~valid_route_mask)),
        "route_disconnected_consecutive_pairs": int(disconnected),
        "route_consecutive_connectivity_pct": float(100 * connected / route_pairs) if route_pairs else np.nan,
        "undirected_components": int(components),
        "isolated_nodes": int(isolated),
        "suggested_utm_epsg": epsg,
    }
    connectivity = {
        "split": split_dir.name,
        "trip": trip_id,
        "route_pairs_checked": route_pairs,
        "connected_pairs": connected,
        "disconnected_pairs": disconnected,
        "route_consecutive_connectivity_pct": inventory["route_consecutive_connectivity_pct"],
    }
    return inventory, connectivity


def create_map(split_dir: Path, trip_id: str, output_path: Path) -> None:
    folder = split_dir / trip_id
    base = folder / trip_id
    track = read_table(Path(str(base) + ".track"))
    nodes = read_table(Path(str(base) + ".nodes"))
    arcs = read_table(Path(str(base) + ".arcs"), dtype=int)
    route = read_table(Path(str(base) + ".route"), dtype=int)

    lon = track.iloc[:, 0].to_numpy(dtype=float)
    lat = track.iloc[:, 1].to_numpy(dtype=float)
    center = [float(np.mean(lat)), float(np.mean(lon))]
    m = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="OpenStreetMap")

    gps_layer = FeatureGroup(name="GPS observations", show=True)
    # Limit marker count for browser performance while retaining the full track as a line.
    step = max(1, len(track) // 1200)
    for i in range(0, len(track), step):
        folium.CircleMarker(
            location=[lat[i], lon[i]], radius=2, color="#1677ff",
            fill=True, fill_opacity=0.75,
            popup=f"GPS observation {i}; t={track.iloc[i, 2]}"
        ).add_to(gps_layer)
    folium.PolyLine(list(zip(lat, lon)), color="#1677ff", weight=2, opacity=0.65,
                    tooltip="GPS trajectory").add_to(gps_layer)
    gps_layer.add_to(m)

    # Draw the road network, with a safety cap to keep the interactive map responsive.
    network_layer = FeatureGroup(name="Road network", show=False)
    node_xy = nodes.to_numpy(dtype=float)
    arc_arr = arcs.to_numpy(dtype=int)
    max_network_arcs = 25000
    network_step = max(1, len(arc_arr) // max_network_arcs)
    for arc_id in range(0, len(arc_arr), network_step):
        u, v = arc_arr[arc_id]
        if 0 <= u < len(node_xy) and 0 <= v < len(node_xy):
            folium.PolyLine(
                [[node_xy[u, 1], node_xy[u, 0]], [node_xy[v, 1], node_xy[v, 0]]],
                color="#777777", weight=1, opacity=0.35
            ).add_to(network_layer)
    network_layer.add_to(m)

    reference_layer = FeatureGroup(name="Reference route", show=True)
    route_ids = route.iloc[:, 0].to_numpy(dtype=int)
    valid = route_ids[(route_ids >= 0) & (route_ids < len(arc_arr))]
    route_points = []
    for arc_id in valid:
        u, v = arc_arr[arc_id]
        if 0 <= u < len(node_xy) and 0 <= v < len(node_xy):
            # Add segment endpoints in route order; avoid duplicate adjacent endpoints.
            for node_id in (u, v):
                point = [float(node_xy[node_id, 1]), float(node_xy[node_id, 0])]
                if not route_points or point != route_points[-1]:
                    route_points.append(point)
    if len(route_points) >= 2:
        folium.PolyLine(route_points, color="#159447", weight=5, opacity=0.9,
                        tooltip="Reference route").add_to(reference_layer)
    reference_layer.add_to(m)

    folium.Marker(center, tooltip=f"Trip {trip_id} centre").add_to(m)
    LayerControl(collapsed=False).add_to(m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help='Path to extracted "Student folder"')
    parser.add_argument("--trip", default="00000034", help="Trip ID to map (default: 00000034)")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out)
    if not data_root.exists():
        raise SystemExit(f"Dataset directory not found: {data_root}")

    inventories, connectivities = [], []
    for split_name in ("train", "student_test"):
        split_dir = data_root / split_name
        if not split_dir.exists():
            print(f"Warning: split not found: {split_dir}")
            continue
        trip_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit())
        for trip_dir in trip_dirs:
            try:
                inv, conn = analyse_trip(split_dir, trip_dir.name)
                inventories.append(inv)
                connectivities.append(conn)
            except Exception as exc:
                print(f"Could not analyse {split_name}/{trip_dir.name}: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_df = pd.DataFrame(inventories)
    connectivity_df = pd.DataFrame(connectivities)
    inventory_df.to_csv(out_dir / "eda_inventory.csv", index=False)
    connectivity_df.to_csv(out_dir / "route_connectivity.csv", index=False)

    summary_lines = [
        "PathSnap EDA summary",
        f"Trips analysed: {len(inventory_df)}",
        f"Training trips: {int((inventory_df['split'] == 'train').sum()) if len(inventory_df) else 0}",
        f"Student-test trips: {int((inventory_df['split'] == 'student_test').sum()) if len(inventory_df) else 0}",
        f"GPS observations total: {int(inventory_df['gps_observations'].sum()) if len(inventory_df) else 0}",
        f"Track NaN cells: {int(inventory_df['track_nan_cells'].sum()) if len(inventory_df) else 0}",
        f"Duplicate GPS rows: {int(inventory_df['duplicate_gps_rows'].sum()) if len(inventory_df) else 0}",
        f"Non-increasing timestamp steps: {int(inventory_df['timestamp_non_increasing_steps'].sum()) if len(inventory_df) else 0}",
        f"Trips with route discontinuities: {int((inventory_df['route_disconnected_consecutive_pairs'] > 0).sum()) if len(inventory_df) else 0}",
        f"Mean route consecutive connectivity (%): {inventory_df['route_consecutive_connectivity_pct'].mean():.3f}" if len(inventory_df) else "Mean route connectivity: N/A",
        "Note: the suggested UTM EPSG is calculated per trip; do not apply one zone to the entire dataset."
    ]
    (out_dir / "eda_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # Prefer the requested trip in training; otherwise use the first available trip.
    map_split = data_root / "train"
    if not (map_split / args.trip).exists():
        map_split = data_root / "student_test"
    if (map_split / args.trip).exists():
        map_path = out_dir / f"trip_{args.trip}_eda_map.html"
        create_map(map_split, args.trip, map_path)
        print(f"Map saved: {map_path}")
    else:
        print(f"Map not created: trip {args.trip} not found in train or student_test.")

    print("\n".join(summary_lines))
    print(f"\\nCSV files saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

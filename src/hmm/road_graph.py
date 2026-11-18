
import heapq
import math

import networkx as nx
from shapely.geometry import Point

from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries


def load_road_graph(trip):
    """
    Build a directed road graph using projected UTM distances.

    Args:
        trip: Dictionary returned by load_trip().

    Returns:
        graph: Directed NetworkX MultiDiGraph.
        arc_lookup: Dictionary containing arc connectivity and lengths.
    """

    nodes = trip["nodes"]
    arcs = trip["arcs"]

    # Project GPS observations and road nodes into UTM coordinates.
    _, nodes_projected, projection = project_trip(
        trip["track"],
        nodes
    )

    # Build road geometries using projected coordinates.
    arc_geometries = build_arc_geometries(
        nodes_projected,
        arcs
    )

    graph = nx.MultiDiGraph()

    # Add road intersection nodes.
    for row in nodes_projected.itertuples(index=False):

        node_id = int(row.node_id)

        graph.add_node(
            node_id,
            x=float(row.x),
            y=float(row.y)
        )

    arc_lookup = {}

    # Add directed road segments.
    for row in arc_geometries.itertuples(index=False):

        arc_id = int(row.arc_id)
        from_node = int(row.from_node)
        to_node = int(row.to_node)

        if from_node not in graph or to_node not in graph:
            raise ValueError(
                f"Arc {arc_id} references invalid nodes: "
                f"{from_node} -> {to_node}"
            )

        # Use projected geometry length in metres.
        distance = float(row.arc_length)

        graph.add_edge(
            from_node,
            to_node,
            key=arc_id,
            arc_id=arc_id,
            length_m=distance
        )

        arc_lookup[arc_id] = {
            "from_node": from_node,
            "to_node": to_node,
            "length_m": distance
        }

    return graph, arc_lookup


def dijkstra_distance(
    graph,
    source,
    target,
    max_distance_m=5000
):
    """
    Find the shortest directed road-network distance.

    Returns infinity if no path exists within the distance cap.
    """

    if source not in graph or target not in graph:
        return math.inf

    if source == target:
        return 0.0

    distances = {
        source: 0.0
    }

    priority_queue = [
        (0.0, source)
    ]

    while priority_queue:

        current_distance, current_node = heapq.heappop(
            priority_queue
        )

        # Ignore outdated queue entries.
        if current_distance > distances[current_node]:
            continue

        # Stop when the distance cap is exceeded.
        if current_distance > max_distance_m:
            break

        if current_node == target:
            return current_distance

        # Explore outgoing directed edges only.
        for _, next_node, _, edge_data in graph.out_edges(
            current_node,
            keys=True,
            data=True
        ):

            edge_length = edge_data["length_m"]

            new_distance = current_distance + edge_length

            if new_distance > max_distance_m:
                continue

            if new_distance < distances.get(next_node, math.inf):

                distances[next_node] = new_distance

                heapq.heappush(
                    priority_queue,
                    (new_distance, next_node)
                )

    return math.inf


def arc_network_distance(
    graph,
    arc_lookup,
    previous_arc,
    current_arc,
    max_distance_m=5000
):
    """
    Calculate connecting distance from the end of the
    previous arc to the start of the current arc.
    """

    if previous_arc not in arc_lookup:
        return math.inf

    if current_arc not in arc_lookup:
        return math.inf

    previous_end = arc_lookup[previous_arc]["to_node"]

    current_start = arc_lookup[current_arc]["from_node"]

    return dijkstra_distance(
        graph,
        previous_end,
        current_start,
        max_distance_m
    )


def partial_arc_distance(
    arc_geometry,
    point_x,
    point_y,
    position="remaining"
):
    """
    Calculate distance along a projected road arc.

    position='initial':
        Distance from arc start to projected GPS position.

    position='remaining':
        Distance from projected GPS position to arc end.
    """

    point = Point(point_x, point_y)

    projected_distance = arc_geometry.project(point)

    if position == "initial":

        return float(projected_distance)

    elif position == "remaining":

        return float(
            arc_geometry.length - projected_distance
        )

    else:

        raise ValueError(
            "position must be 'initial' or 'remaining'"
        )


def complete_network_distance(
    graph,
    arc_lookup,
    previous_arc,
    current_arc,
    previous_geometry,
    current_geometry,
    previous_x,
    previous_y,
    current_x,
    current_y,
    max_distance_m=5000
):
    """
    Calculate network distance between two consecutive
    GPS observations assigned to candidate road arcs.

    Handles:
        1. Same directed arc.
        2. Different directed arcs.

    Distance for different arcs =
        remaining previous arc
        + connecting shortest path
        + initial current arc.
    """

    if previous_arc not in arc_lookup:
        return math.inf

    if current_arc not in arc_lookup:
        return math.inf

    # Project both GPS positions onto their candidate arcs.
    previous_position = previous_geometry.project(
        Point(previous_x, previous_y)
    )

    current_position = current_geometry.project(
        Point(current_x, current_y)
    )

    # CASE 1: Both observations belong to the same directed arc.
    if previous_arc == current_arc:

        distance = current_position - previous_position

        if 0 <= distance <= max_distance_m:
            return float(distance)

        return math.inf

    # CASE 2: Observations belong to different directed arcs.

    # Remaining distance along previous arc.
    remaining_previous = (
        previous_geometry.length - previous_position
    )

    # Shortest directed path between the two arcs.
    connecting_distance = arc_network_distance(
        graph,
        arc_lookup,
        previous_arc,
        current_arc,
        max_distance_m
    )

    if not math.isfinite(connecting_distance):
        return math.inf

    # Initial distance along current arc.
    initial_current = current_position

    total_distance = (
        remaining_previous
        + connecting_distance
        + initial_current
    )

    if total_distance > max_distance_m:
        return math.inf

    return float(total_distance)

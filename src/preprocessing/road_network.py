
import numpy as np
from shapely.geometry import LineString


def build_arc_geometries(nodes, arcs):
    """
    Build a LineString for every road arc.

    arc_id remains the original row index
    from the .arcs file.
    """

    node_xy = nodes.set_index("node_id")[["x", "y"]]

    geometries = []

    for row in arcs.itertuples(index=False):

        start = node_xy.loc[row.from_node]
        end = node_xy.loc[row.to_node]

        geometry = LineString([
            (start["x"], start["y"]),
            (end["x"], end["y"])
        ])

        geometries.append(geometry)

    result = arcs.copy()

    result["geometry"] = geometries
    result["arc_length"] = [
        geometry.length for geometry in geometries
    ]

    return result


def build_spatial_index(arc_geometries):
    """
    Build a Shapely STRtree for efficient spatial queries.
    """

    from shapely import STRtree

    geometries = arc_geometries["geometry"].tolist()

    tree = STRtree(geometries)

    return tree

import numpy as np
from pyproj import CRS, Transformer


def get_utm_crs(lon, lat):
    """
    Determine the appropriate UTM CRS from longitude
    and latitude.

    Returns an EPSG code.
    """

    lon = float(lon)
    lat = float(lat)

    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon}")

    if not (-80 <= lat <= 84):
        raise ValueError(
            "UTM is not appropriate outside latitudes -80 to 84."
        )

    zone = int(np.floor((lon + 180) / 6)) + 1
    zone = min(max(zone, 1), 60)

    epsg = (32600 if lat >= 0 else 32700) + zone

    return CRS.from_epsg(epsg)


def project_trip(track, nodes):
    """
    Convert GPS observations and road nodes
    from longitude/latitude to projected metres.

    The same CRS is used for both datasets
    belonging to a trip.
    """

    if track.empty or nodes.empty:
        raise ValueError("Track and nodes cannot be empty.")

    # Use median GPS location to select the local UTM zone.
    center_lon = track["lon"].median()
    center_lat = track["lat"].median()

    crs = get_utm_crs(center_lon, center_lat)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    track_projected = track.copy()
    nodes_projected = nodes.copy()

    track_projected["x"], track_projected["y"] = (
        transformer.transform(
            track["lon"].to_numpy(),
            track["lat"].to_numpy()
        )
    )

    nodes_projected["x"], nodes_projected["y"] = (
        transformer.transform(
            nodes["lon"].to_numpy(),
            nodes["lat"].to_numpy()
        )
    )

    metadata = {
        "epsg": crs.to_epsg(),
        "crs": crs.to_string(),
        "center_lon": center_lon,
        "center_lat": center_lat,
    }

    return track_projected, nodes_projected, metadata
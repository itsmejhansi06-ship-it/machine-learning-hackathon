
import math
from pyproj import Geod

GEOD = Geod(ellps="WGS84")


def gps_distance_m(lon1, lat1, lon2, lat2):
    """Calculate straight-line GPS distance in metres."""
    _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(distance)


def transition_probability(
    network_distance,
    gps_distance,
    beta=50.0,
    min_probability=1e-12
):
    """Calculate the transition probability between two candidate arcs."""

    if beta <= 0:
        raise ValueError("Beta must be positive")

    if (
        not math.isfinite(network_distance)
        or network_distance < 0
        or not math.isfinite(gps_distance)
        or gps_distance < 0
    ):
        return min_probability

    difference = abs(network_distance - gps_distance)

    probability = math.exp(-difference / beta)

    return max(probability, min_probability)


def log_transition_probability(
    network_distance,
    gps_distance,
    beta=50.0
):
    """Return log transition probability for Viterbi."""
    probability = transition_probability(
        network_distance,
        gps_distance,
        beta
    )

    return math.log(probability)

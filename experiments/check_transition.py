
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.hmm.transition import (
    gps_distance_m,
    transition_probability,
    log_transition_probability
)

# Two GPS observations
lon1, lat1 = 8.581679, 49.857880
lon2, lat2 = 8.584637, 49.858295

# Calculate straight-line GPS distance
gps_distance = gps_distance_m(lon1, lat1, lon2, lat2)

# Example network distance
network_distance = 350.0

# Calculate transition probability
probability = transition_probability(
    network_distance,
    gps_distance,
    beta=50.0
)

log_probability = log_transition_probability(
    network_distance,
    gps_distance,
    beta=50.0
)

print("TRANSITION MODEL TEST")
print("---------------------")
print(f"GPS distance: {gps_distance:.2f} m")
print(f"Network distance: {network_distance:.2f} m")
print(f"Transition probability: {probability:.10f}")
print(f"Log transition probability: {log_probability:.6f}")

# Test disconnected roads
disconnected_probability = transition_probability(
    float("inf"),
    gps_distance,
    beta=50.0
)

print("\nDISCONNECTED ROAD TEST")
print(f"Probability: {disconnected_probability}")

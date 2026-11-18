
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.hmm.emission import (
    gaussian_emission_probability,
    log_emission_probability
)

# Test different distances from a GPS observation to a road
distances = [0, 10, 30, 60, 100]

print("GAUSSIAN EMISSION TEST")
print("----------------------")

for distance in distances:
    probability = gaussian_emission_probability(
        distance,
        sigma=30.0
    )

    log_probability = log_emission_probability(
        distance,
        sigma=30.0
    )

    print(
        f"Distance: {distance:3} m | "
        f"Probability: {probability:.10f} | "
        f"Log probability: {log_probability:.6f}"
    )

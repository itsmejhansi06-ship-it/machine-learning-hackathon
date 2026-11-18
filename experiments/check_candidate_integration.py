
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.preprocessing.candidates import build_candidates

TRIP_ID = "00000001"

# Load the trip
trip = load_trip(TRIP_ID, split="train")

# Generate candidate road arcs
candidates = build_candidates(trip)

print("CANDIDATE INTEGRATION TEST")
print("--------------------------")

print("Trip ID:", trip["trip_id"])
print("GPS observations:", len(trip["track"]))
print("Candidate rows:", len(candidates))

print("\nColumns:")
print(candidates.columns.tolist())

print("\nFirst 5 candidate rows:")
print(candidates.head())

# Check required columns
required_columns = {
    "trip",
    "obs_idx",
    "t",
    "arc_id",
    "distance",
    "distance_rank",
    "arc_length"
}

assert required_columns.issubset(candidates.columns)

# Check candidate availability
available_observations = candidates["obs_idx"].nunique()

print("\nObservations with candidates:", available_observations)
print("Total observations:", len(trip["track"]))

print("\nCandidate integration test completed!")

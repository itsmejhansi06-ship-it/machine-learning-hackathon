
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.hmm.hmm import map_match_trip


def main():

    print("\nHMM PIPELINE TEST - FULL TRIP")
    print("=" * 45)

    # Load original training trip
    trip = load_trip("00000001", split="train")

    print("Trip ID:", trip["trip_id"])
    print("Original observations:", len(trip["track"]))

    # Use all observations
    full_trip = trip.copy()
    full_trip["track"] = trip["track"].copy()

    print("Test observations:", len(full_trip["track"]))

    print("\nRunning HMM...")
    start_time = time.time()

    predictions = map_match_trip(
        full_trip,
        k=5,
        radius=50.0
    )

    elapsed = time.time() - start_time

    print("\nPREDICTION RESULTS")
    print("-" * 45)

    print("Predicted observations:", len(predictions))
    print("Execution time:", round(elapsed, 2), "seconds")

    print("\nFirst 10 predictions:")
    print(predictions.head(10).to_string(index=False))

    # Basic validation
    assert len(predictions) == len(full_trip["track"])
    assert predictions["predicted_arc_id"].notna().all()

    print("\nUnique predicted arcs:",
          predictions["predicted_arc_id"].nunique())

    print("\nSUCCESS: Full HMM pipeline test passed!")


if __name__ == "__main__":
    main()

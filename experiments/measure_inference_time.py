
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.preprocessing.candidates import build_candidates
from src.hmm.hmm import map_match_trip


def main():
    # Load training trip
    trip = load_trip("00000001", split="train")

    # Generate candidates only once
    print("Preparing candidates...", flush=True)
    candidates = build_candidates(trip)

    print("Starting HMM inference...", flush=True)

    # Measure one HMM execution
    start = time.perf_counter()

    predictions = map_match_trip(
        trip,
        cand_df=candidates
    )

    end = time.perf_counter()

    elapsed = end - start

    # Display results
    observations = len(trip["track"])

    print("\nINFERENCE TIME RESULTS")
    print("=" * 45)

    print("GPS observations:", observations)
    print("Candidate rows:", len(candidates))
    print("Predictions:", len(predictions))

    print(f"HMM inference time: {elapsed:.4f} seconds")

    print(
        f"Average time per observation: "
        f"{elapsed / observations * 1000:.4f} ms"
    )


if __name__ == "__main__":
    main()

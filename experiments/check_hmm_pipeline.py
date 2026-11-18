
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.hmm.hmm import map_match_trip


def main():

    print("\nHMM PIPELINE TEST")
    print("=" * 45)

    # Load training trip
    trip = load_trip("00000001", split="train")

    print("Trip ID:", trip["trip_id"])
    print("GPS observations:", len(trip["track"]))

    # Run HMM map matching
    predictions = map_match_trip(trip)

    print("\nPREDICTION RESULTS")
    print("-" * 45)

    print("Predicted observations:", len(predictions))

    print("\nFirst 10 predictions:")
    print(predictions.head(10).to_string(index=False))

    print("\nLast 10 predictions:")
    print(predictions.tail(10).to_string(index=False))

    # Basic validation
    assert len(predictions) == len(trip["track"])

    assert predictions["predicted_arc_id"].notna().all()

    print("\nUnique predicted arcs:",
          predictions["predicted_arc_id"].nunique())

    print("\nSUCCESS: HMM pipeline produced predictions!")


if __name__ == "__main__":
    main()

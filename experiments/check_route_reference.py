
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.hmm.hmm import map_match_trip


def main():

    trip = load_trip("00000001", split="train")

    predictions = map_match_trip(trip)

    reference_route = trip["route"]

    predicted_route = predictions[
        "predicted_arc_id"
    ].tolist()

    # Remove consecutive duplicate arcs.
    compressed_predictions = []

    for arc in predicted_route:
        if (
            not compressed_predictions
            or compressed_predictions[-1] != arc
        ):
            compressed_predictions.append(arc)

    print("\nROUTE EVALUATION INSPECTION")
    print("=" * 45)

    print("GPS observations:", len(trip["track"]))
    print("Reference route length:", len(reference_route))
    print("Predicted observation sequence:", len(predicted_route))
    print("Compressed predicted route:", len(compressed_predictions))

    print("\nFirst 20 reference arc IDs:")
    print(reference_route[:20])

    print("\nFirst 20 predicted arc IDs:")
    print(compressed_predictions[:20])

    print("\nReference unique arcs:", len(set(reference_route)))
    print("Predicted unique arcs:", len(set(compressed_predictions)))


if __name__ == "__main__":
    main()

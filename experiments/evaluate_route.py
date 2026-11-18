import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_trip
from src.hmm.hmm import map_match_trip


def longest_common_subsequence(a, b):
    m, n = len(a), len(b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1]
                )

    return dp[m][n]


def main():
    trip = load_trip("00000001", split="train")

    predictions = map_match_trip(trip)

    reference = trip["route"].tolist()

    predicted = predictions["predicted_arc_id"].tolist()

    compressed = []

    for arc in predicted:
        if not compressed or compressed[-1] != arc:
            compressed.append(arc)

    lcs_length = longest_common_subsequence(
        reference,
        compressed
    )

    recall = lcs_length / len(reference)

    precision = lcs_length / len(compressed)

    print("\nROUTE EVALUATION")
    print("=" * 40)

    print("Reference route length:", len(reference))
    print("Predicted route length:", len(compressed))
    print("Matching ordered arcs:", lcs_length)

    print(f"Route recall: {recall:.4f}")
    print(f"Route precision: {precision:.4f}")


if __name__ == "__main__":
    main()
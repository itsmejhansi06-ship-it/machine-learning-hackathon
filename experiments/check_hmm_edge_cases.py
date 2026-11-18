
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.hmm.viterbi import viterbi


def main():

    print("HMM EDGE CASE TESTING")
    print("=" * 40)

    # Test 1: Empty observations
    result = viterbi([], [])
    assert result == []
    print("Test 1 passed: Empty observations")

    # Test 2: Single observation
    emissions = [
        {10: -1.0, 20: -3.0}
    ]

    result = viterbi(emissions, [])

    assert result == [10]
    print("Test 2 passed: Single observation")

    # Test 3: Multiple observations
    emissions = [
        {10: -1.0, 20: -2.0},
        {30: -1.0, 40: -3.0}
    ]

    transitions = [
        {
            (10, 30): -0.1,
            (10, 40): -2.0,
            (20, 30): -3.0,
            (20, 40): -0.1
        }
    ]

    result = viterbi(emissions, transitions)

    assert len(result) == 2
    print("Test 3 passed: Multiple observations")

    # Test 4: Missing transition probabilities
    emissions = [
        {10: -1.0},
        {20: -1.0}
    ]

    transitions = [{}]

    result = viterbi(emissions, transitions)

    assert result == [10, 20]
    print("Test 4 passed: Missing transition probability")

    # Test 5: Invalid transition list length
    try:
        viterbi(
            [{10: -1.0}, {20: -1.0}],
            []
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        print("Test 5 passed: Invalid transition list length")

    print("\nAll edge-case tests passed!")


if __name__ == "__main__":
    main()

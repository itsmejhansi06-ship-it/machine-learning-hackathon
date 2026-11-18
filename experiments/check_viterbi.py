
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.hmm.viterbi import viterbi

import math

# Three possible road arcs at each GPS observation
states = [10, 11, 12]

# Emission log probabilities for three observations
emission_log_probs = [
    {
        10: math.log(0.8),
        11: math.log(0.15),
        12: math.log(0.05)
    },
    {
        10: math.log(0.1),
        11: math.log(0.7),
        12: math.log(0.2)
    },
    {
        10: math.log(0.1),
        11: math.log(0.2),
        12: math.log(0.7)
    }
]

# Transition log probabilities between consecutive observations
transition_log_probs = [
    {
        (10, 10): math.log(0.1),
        (10, 11): math.log(0.8),
        (10, 12): math.log(0.1),

        (11, 10): math.log(0.1),
        (11, 11): math.log(0.8),
        (11, 12): math.log(0.1),

        (12, 10): math.log(0.1),
        (12, 11): math.log(0.2),
        (12, 12): math.log(0.7)
    },
    {
        (10, 10): math.log(0.1),
        (10, 11): math.log(0.2),
        (10, 12): math.log(0.7),

        (11, 10): math.log(0.1),
        (11, 11): math.log(0.2),
        (11, 12): math.log(0.7),

        (12, 10): math.log(0.1),
        (12, 11): math.log(0.2),
        (12, 12): math.log(0.7)
    }
]

# Run Viterbi
best_path = viterbi(
    emission_log_probs,
    transition_log_probs,
    states
)

print("VITERBI TEST")
print("------------")
print("Best predicted arc sequence:", best_path)

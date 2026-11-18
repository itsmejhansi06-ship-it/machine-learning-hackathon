
import math


def viterbi(emission_log_probs, transition_log_probs, states=None):
    """
    Find the most likely hidden-state sequence using Viterbi.

    emission_log_probs:
        List of dictionaries.
        Each dictionary maps candidate arc IDs to log emission
        probabilities for one GPS observation.

    transition_log_probs:
        List of dictionaries.
        Each dictionary maps (previous_arc, current_arc) to
        log transition probabilities between consecutive observations.

    states:
        Optional list of possible states.
        If omitted, states are obtained from emission_log_probs.

    Returns:
        Most likely sequence of candidate arc IDs.
    """

    if not emission_log_probs:
        return []

    if len(transition_log_probs) != len(emission_log_probs) - 1:
        raise ValueError(
            "There must be one transition dictionary "
            "between every pair of consecutive observations."
        )

    # Get candidate states separately for each observation.
    if states is None:
        states_by_time = [
            list(emissions.keys())
            for emissions in emission_log_probs
        ]
    else:
        states_by_time = [
            list(states)
            for _ in emission_log_probs
        ]

    # Validate state availability.
    if any(not current_states for current_states in states_by_time):
        raise ValueError(
            "Every observation must have at least one candidate state."
        )

    # Initialization
    scores = {}

    for state in states_by_time[0]:
        scores[state] = emission_log_probs[0].get(
            state,
            math.log(1e-12)
        )

    backpointers = []

    # Recursion
    for t in range(1, len(emission_log_probs)):

        new_scores = {}
        backpointer = {}

        for current_state in states_by_time[t]:

            best_score = -math.inf
            best_previous_state = None

            emission = emission_log_probs[t].get(
                current_state,
                math.log(1e-12)
            )

            for previous_state in states_by_time[t - 1]:

                transition = transition_log_probs[t - 1].get(
                    (previous_state, current_state),
                    math.log(1e-12)
                )

                score = (
                    scores[previous_state]
                    + transition
                    + emission
                )

                if score > best_score:
                    best_score = score
                    best_previous_state = previous_state

            new_scores[current_state] = best_score
            backpointer[current_state] = best_previous_state

        scores = new_scores
        backpointers.append(backpointer)

    # Termination
    best_final_state = max(scores, key=scores.get)

    # Backtracking
    best_path = [best_final_state]

    for backpointer in reversed(backpointers):
        best_path.append(
            backpointer[best_path[-1]]
        )

    best_path.reverse()

    return best_path

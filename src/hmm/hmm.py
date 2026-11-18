
import math
import time

import pandas as pd
from shapely.geometry import Point

from src.preprocessing.candidates import build_candidates
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries

from src.hmm.road_graph import (
    load_road_graph,
    arc_network_distance,
)

from src.hmm.emission import log_emission_probability

from src.hmm.transition import (
    gps_distance_m,
    log_transition_probability,
)

from src.hmm.viterbi import viterbi


def map_match_trip(
    trip,
    cand_df=None,
    k=10,
    radius=50.0,
    sigma=30.0,
    beta=50.0,
    max_distance_m=5000
):
    """
    Perform HMM-based map matching for one GPS trip.

    Returns:
        DataFrame containing obs_idx and predicted_arc_id.
    """

    total_start = time.time()

    # --------------------------------------------------
    # STEP 1: Generate candidates
    # --------------------------------------------------

    print("\n[HMM] Generating candidates...", flush=True)

    if cand_df is None:
        cand_df = build_candidates(
            trip,
            k=k,
            radius=radius
        )

    if cand_df.empty:
        raise ValueError("No candidate road arcs were generated.")

    cand_df = cand_df.sort_values(
        ["obs_idx", "distance_rank"]
    ).copy()

    track = trip["track"].sort_values("obs_idx").copy()

    observation_ids = track["obs_idx"].astype(int).tolist()

    candidate_observations = set(
        cand_df["obs_idx"].astype(int)
    )

    missing_observations = [
        obs_id
        for obs_id in observation_ids
        if obs_id not in candidate_observations
    ]

    if missing_observations:
        raise ValueError(
            f"No candidates found for observations: "
            f"{missing_observations[:10]}"
        )

    print(
        f"[HMM] Observations: {len(observation_ids)}, "
        f"candidate rows: {len(cand_df)}",
        flush=True
    )

    # --------------------------------------------------
    # STEP 2: Prepare candidate states
    # --------------------------------------------------

    print("[HMM] Preparing candidate states...", flush=True)

    states_by_time = []
    candidates_by_time = []

    for obs_id in observation_ids:

        observation_candidates = cand_df[
            cand_df["obs_idx"] == obs_id
        ]

        states = (
            observation_candidates["arc_id"]
            .astype(int)
            .tolist()
        )

        states_by_time.append(states)
        candidates_by_time.append(observation_candidates)

    # --------------------------------------------------
    # STEP 3: Calculate emission probabilities
    # --------------------------------------------------

    print("[HMM] Calculating emission probabilities...", flush=True)

    emission_log_probs = []

    for observation_candidates in candidates_by_time:

        emission_dict = {}

        for row in observation_candidates.itertuples(index=False):

            arc_id = int(row.arc_id)
            distance = float(row.distance)

            emission_dict[arc_id] = log_emission_probability(
                distance,
                sigma=sigma
            )

        emission_log_probs.append(emission_dict)

    # --------------------------------------------------
    # STEP 4: Build projected road geometries and graph
    # --------------------------------------------------

    print("[HMM] Building road graph and geometries...", flush=True)

    track_projected, nodes_projected, projection = project_trip(
        trip["track"],
        trip["nodes"]
    )

    arc_geometries = build_arc_geometries(
        nodes_projected,
        trip["arcs"]
    )

    geometry_lookup = {
        int(row.arc_id): row.geometry
        for row in arc_geometries.itertuples(index=False)
    }

    graph, arc_lookup = load_road_graph(trip)

    gps_lookup = {
        int(row.obs_idx): row
        for row in track.itertuples(index=False)
    }

    projected_gps_lookup = {
        int(row.obs_idx): row
        for row in track_projected.itertuples(index=False)
    }

    # --------------------------------------------------
    # STEP 5: Precompute candidate positions
    # --------------------------------------------------

    print("[HMM] Precomputing candidate positions...", flush=True)

    position_lookup = {}

    for t, obs_id in enumerate(observation_ids):

        projected_gps = projected_gps_lookup[obs_id]

        gps_x = float(projected_gps.x)
        gps_y = float(projected_gps.y)

        position_dict = {}

        for arc_id in states_by_time[t]:

            geometry = geometry_lookup[arc_id]

            position_dict[arc_id] = geometry.project(
                Point(gps_x, gps_y)
            )

        position_lookup[obs_id] = position_dict

    # --------------------------------------------------
    # STEP 6: Calculate transition probabilities
    # --------------------------------------------------

    print("[HMM] Calculating transition probabilities...", flush=True)

    transition_start = time.time()

    transition_log_probs = []

    connecting_distance_cache = {}

    total_transitions = len(observation_ids) - 1

    for t in range(1, len(observation_ids)):

        previous_obs_id = observation_ids[t - 1]
        current_obs_id = observation_ids[t]

        previous_gps = gps_lookup[previous_obs_id]
        current_gps = gps_lookup[current_obs_id]

        straight_distance = gps_distance_m(
            float(previous_gps.lon),
            float(previous_gps.lat),
            float(current_gps.lon),
            float(current_gps.lat)
        )

        previous_positions = position_lookup[previous_obs_id]
        current_positions = position_lookup[current_obs_id]

        transition_dict = {}

        for previous_arc in states_by_time[t - 1]:

            previous_position = previous_positions[previous_arc]
            previous_geometry = geometry_lookup[previous_arc]

            for current_arc in states_by_time[t]:

                current_position = current_positions[current_arc]
                current_geometry = geometry_lookup[current_arc]

                # CASE 1: Same directed arc
                if previous_arc == current_arc:

                    network_distance = (
                        current_position - previous_position
                    )

                    if not (
                        0 <= network_distance <= max_distance_m
                    ):
                        network_distance = math.inf

                # CASE 2: Different directed arcs
                else:

                    cache_key = (
                        previous_arc,
                        current_arc
                    )

                    if cache_key not in connecting_distance_cache:

                        connecting_distance_cache[cache_key] = (
                            arc_network_distance(
                                graph,
                                arc_lookup,
                                previous_arc,
                                current_arc,
                                max_distance_m
                            )
                        )

                    connecting_distance = (
                        connecting_distance_cache[cache_key]
                    )

                    if not math.isfinite(connecting_distance):

                        network_distance = math.inf

                    else:

                        remaining_previous = (
                            previous_geometry.length
                            - previous_position
                        )

                        initial_current = current_position

                        network_distance = (
                            remaining_previous
                            + connecting_distance
                            + initial_current
                        )

                        if network_distance > max_distance_m:
                            network_distance = math.inf

                transition_log = log_transition_probability(
                    network_distance,
                    straight_distance,
                    beta=beta
                )

                transition_dict[
                    (previous_arc, current_arc)
                ] = transition_log

        transition_log_probs.append(transition_dict)

        # Print progress every 10 observations.
        if t % 10 == 0 or t == total_transitions:

            print(
                f"[HMM] Transition progress: "
                f"{t}/{total_transitions} | "
                f"Unique arc pairs cached: "
                f"{len(connecting_distance_cache)}",
                flush=True
            )

    transition_elapsed = time.time() - transition_start

    print("\n[HMM] Transition calculation completed.", flush=True)

    print(
        "[HMM] Unique arc pairs cached:",
        len(connecting_distance_cache),
        flush=True
    )

    print(
        "[HMM] Transition calculation time:",
        round(transition_elapsed, 2),
        "seconds",
        flush=True
    )

    # --------------------------------------------------
    # STEP 7: Run Viterbi
    # --------------------------------------------------

    print("\n[HMM] Running Viterbi...", flush=True)

    predicted_path = viterbi(
        emission_log_probs=emission_log_probs,
        transition_log_probs=transition_log_probs,
        states=None
    )

    # --------------------------------------------------
    # STEP 8: Return predictions
    # --------------------------------------------------

    predictions = pd.DataFrame({
        "obs_idx": observation_ids,
        "predicted_arc_id": predicted_path
    })

    total_elapsed = time.time() - total_start

    print("\n[HMM] Map matching completed.", flush=True)

    print(
        "[HMM] Total execution time:",
        round(total_elapsed, 2),
        "seconds",
        flush=True
    )

    return predictions


import pandas as pd


def attach_candidate_labels(candidates, alignment):
    """
    Attach reference-route labels to candidate arcs.

    label = 1: candidate arc equals the aligned reference arc
    label = 0: candidate arc does not equal the aligned reference arc
    """

    required_candidates = {"obs_idx", "arc_id"}
    required_alignment = {
        "obs_idx",
        "reference_arc_id",
        "alignment_distance",
        "route_position",
    }

    if not required_candidates.issubset(candidates.columns):
        raise ValueError("Candidate table is missing required columns.")

    if not required_alignment.issubset(alignment.columns):
        raise ValueError("Alignment table is missing required columns.")

    if alignment["obs_idx"].duplicated().any():
        raise ValueError("Alignment contains duplicate observation IDs.")

    alignment_columns = [
        "obs_idx",
        "reference_arc_id",
        "alignment_distance",
        "route_position",
    ]

    if candidates.empty:
        labelled = candidates.copy()
        for column in alignment_columns[1:]:
            labelled[column] = pd.Series(dtype="float64")
        labelled["label"] = pd.Series(dtype="int64")
    else:
        labelled = candidates.merge(
            alignment[alignment_columns],
            on="obs_idx",
            how="left",
            validate="many_to_one",
        )

        if labelled["reference_arc_id"].isna().any():
            raise ValueError(
                "Some candidate observations have no reference alignment."
            )

        labelled["label"] = (
            labelled["arc_id"] == labelled["reference_arc_id"]
        ).astype("int64")

    candidate_hits = (
        labelled.groupby("obs_idx")["label"].max()
        if not labelled.empty
        else pd.Series(dtype="int64")
    )

    aligned_ids = set(alignment["obs_idx"])
    hit_ids = set(candidate_hits[candidate_hits == 1].index)
    missed_ids = sorted(aligned_ids - hit_ids)

    total = len(alignment)
    misses = len(missed_ids)

    miss_rate = misses / total if total else float("nan")

    missed_details = alignment[
        alignment["obs_idx"].isin(missed_ids)
    ].copy()

    summary = {
        "total_observations": total,
        "observations_with_correct_candidate": total - misses,
        "observations_missing_correct_candidate": misses,
        "candidate_miss_rate": miss_rate,
        "missed_observation_ids": missed_ids,
        "missed_observation_details": missed_details,
    }

    return labelled, summary

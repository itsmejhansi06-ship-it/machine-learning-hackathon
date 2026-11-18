
import argparse
import ast
from pathlib import Path

import pandas as pd

from src.data_loader import get_trip_ids, load_trip
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.preprocessing.candidates import build_candidates
from src.preprocessing.route_alignment import align_route_to_track
from src.features.labels import attach_candidate_labels
from src.features.feature_engineering import add_candidate_features


def process_trip(trip_id, split, k, radius):
    """Run the complete Person B pipeline for one trip."""

    print(f"\nProcessing {split}/{trip_id}...")

    trip = load_trip(trip_id, split=split)

    # Project GPS observations and road nodes into metres.
    track_projected, nodes_projected, projection = project_trip(
        trip["track"],
        trip["nodes"]
    )

    # Build directed road-arc geometries.
    arc_geometries = build_arc_geometries(
        nodes_projected,
        trip["arcs"]
    )

    # Generate candidate road segments.
    candidates = build_candidates(
        trip,
        k=k,
        radius=radius
    )

    # Align the ordered reference route to GPS observations.
    alignment = align_route_to_track(
        track_projected,
        arc_geometries,
        trip["route"]
    )

    # Attach candidate labels and calculate candidate miss rate.
    labelled, summary = attach_candidate_labels(
        candidates,
        alignment
    )

    # Generate engineered features.
    featured = add_candidate_features(
        labelled,
        track_projected,
        arc_geometries
    )

    # Do not export target-derived alignment information as model
    # input features. The label is the supervised target.
    leakage_columns = [
        "reference_arc_id",
        "alignment_distance",
        "route_position"
    ]

    featured = featured.drop(
        columns=leakage_columns,
        errors="ignore"
    )

    # Add dataset and projection metadata.
    featured["split"] = split
    featured["projection_epsg"] = projection["epsg"]

    summary_row = {
        "trip": str(trip_id),
        "split": split,
        "total_observations": summary["total_observations"],
        "observations_with_correct_candidate":
            summary["observations_with_correct_candidate"],
        "observations_missing_correct_candidate":
            summary["observations_missing_correct_candidate"],
        "candidate_miss_rate": summary["candidate_miss_rate"],
        "candidate_rows": len(featured),
        "positive_candidates": (
            int(featured["label"].sum()) if not featured.empty else 0
        ),
        "negative_candidates": (
            int((featured["label"] == 0).sum())
            if not featured.empty else 0
        ),
        "k": k,
        "radius_m": radius,
        "projection_epsg": projection["epsg"],
        "missed_observation_ids": str(
            summary["missed_observation_ids"]
        )
    }

    print(
        f"GPS observations: {summary['total_observations']}"
    )
    print(f"Candidate rows: {len(featured)}")
    print(
        f"Candidate miss rate: "
        f"{summary['candidate_miss_rate']:.4%}"
    )

    return featured, summary_row


def process_split(split, k, radius):
    """Process every trip in one permitted development split."""

    trip_ids = get_trip_ids(split=split)

    print(f"\nFound {len(trip_ids)} trips in {split}.")

    all_features = []
    all_summaries = []

    for index, trip_id in enumerate(trip_ids, start=1):
        print(f"\nTrip {index}/{len(trip_ids)}")

        features, summary = process_trip(
            trip_id,
            split,
            k,
            radius
        )

        all_features.append(features)
        all_summaries.append(summary)

    if all_features:
        combined = pd.concat(
            all_features,
            ignore_index=True
        )
    else:
        combined = pd.DataFrame()

    return combined, all_summaries


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--radius", type=float, default=50.0)

    args = parser.parse_args()

    if args.k < 1 or args.radius <= 0:
        raise ValueError("K and radius must be positive.")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process only permitted development datasets.
    train_features, train_summaries = process_split(
        "train",
        args.k,
        args.radius
    )

    student_features, student_summaries = process_split(
        "student_test",
        args.k,
        args.radius
    )

    # Save combined feature datasets.
    train_path = output_dir / "train_features.parquet"
    student_path = output_dir / "student_test_features.parquet"

    train_features.to_parquet(
        train_path,
        index=False
    )

    student_features.to_parquet(
        student_path,
        index=False
    )

    # Save per-trip candidate miss-rate information.
    summaries = train_summaries + student_summaries

    summary_df = pd.DataFrame(summaries)

    summary_path = output_dir / "candidate_miss_rates.csv"

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n========== DATASET BUILD COMPLETE ==========")
    print(f"Training rows: {len(train_features)}")
    print(f"Student-test rows: {len(student_features)}")
    print(f"Training trips: {len(train_summaries)}")
    print(f"Student-test trips: {len(student_summaries)}")

    print("\nOverall candidate miss rates:")

    for split, group in summary_df.groupby("split"):
        total = group["total_observations"].sum()
        misses = group[
            "observations_missing_correct_candidate"
        ].sum()

        rate = misses / total if total else float("nan")

        print(
            f"{split}: {misses}/{total} = {rate:.4%}"
        )

    print("\nSaved:")
    print(train_path)
    print(student_path)
    print(summary_path)


if __name__ == "__main__":
    main()

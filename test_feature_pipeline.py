
from src.data_loader import load_trip, get_trip_ids
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.preprocessing.candidates import build_candidates
from src.preprocessing.route_alignment import align_route_to_track
from src.features.labels import attach_candidate_labels
from src.features.feature_engineering import add_candidate_features


def test_feature_pipeline(trip_id, split):
    print("\n" + "=" * 65)
    print(f"Testing feature pipeline: {trip_id} ({split})")
    print("=" * 65)

    trip = load_trip(trip_id, split=split)

    # 1. Project GPS coordinates and road nodes.
    track, nodes, projection = project_trip(
        trip["track"], trip["nodes"]
    )

    # 2. Construct road geometries.
    arc_geometries = build_arc_geometries(
        nodes, trip["arcs"]
    )

    # 3. Generate candidates.
    candidates = build_candidates(
        trip, k=30, radius=50
    )

    # 4. Align GPS observations with the reference route.
    alignment = align_route_to_track(
        track, arc_geometries, trip["route"]
    )

    print("\nFirst 20 GPS observations and their aligned reference arcs:")

    alignment_check = alignment.merge(
        trip["track"][["obs_idx", "lon", "lat"]],
        on="obs_idx",
        how="left"
    )

    print(
        alignment_check[
            [
                "obs_idx",
                "reference_arc_id",
                "alignment_distance",
                "route_position",
                "lon",
                "lat"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nFirst 20 reference route arcs:")
    print(trip["route"][:20])

    # 5. Attach labels.
    labelled, summary = attach_candidate_labels(
        candidates, alignment
    )

    # 6. Generate features.
    features = add_candidate_features(
        labelled, track, arc_geometries
    )


    # Investigate observations whose reference arc is missing
    missed_ids = summary["missed_observation_ids"]

    if missed_ids:
        print("\nMISSED CANDIDATE INVESTIGATION")

        for obs_id in missed_ids:
            reference = alignment.loc[
                alignment["obs_idx"] == obs_id
            ].iloc[0]

            available = candidates.loc[
                candidates["obs_idx"] == obs_id
            ].sort_values("distance")

            print(f"\nObservation: {obs_id}")
            print("Reference arc:", reference["reference_arc_id"])
            print(
                "Alignment distance:",
                round(reference["alignment_distance"], 3), "m"
            )

            print("Available candidates:")
            print(
                available[
                    ["arc_id", "distance", "distance_rank"]
                ].to_string(index=False)
            )

            reference_geometry = arc_geometries.loc[
                arc_geometries["arc_id"]
                == reference["reference_arc_id"],
                "geometry"
            ].iloc[0]

            point_row = track.loc[
                track["obs_idx"] == obs_id
            ].iloc[0]

            from shapely.geometry import Point

            reference_distance = Point(
                point_row["x"], point_row["y"]
            ).distance(reference_geometry)

            print(
                "Actual distance to reference arc:",
                round(reference_distance, 3), "m"
            )


    print("\nProjection EPSG:", projection["epsg"])
    print("GPS observations:", len(track))
    print("Candidate rows:", len(candidates))
    print("Labelled rows:", len(labelled))
    print("Feature rows:", len(features))

    print("\nCandidate miss rate:",
          f"{summary['candidate_miss_rate'] * 100:.3f}%")

    print("Missed observation IDs:",
          summary["missed_observation_ids"])

    if not summary["missed_observation_details"].empty:
        print("\nMissed observation details:")
        print(
            summary["missed_observation_details"].to_string(
                index=False
            )
        )

    feature_columns = [
        "distance",
        "distance_rank",
        "arc_length",
        "gps_speed_mps",
        "gps_heading_deg",
        "arc_heading_deg",
        "heading_diff_deg",
        "projection_fraction",
        "along_arc_distance",
        "cross_track_distance",
        "label",
    ]

    print("\nFeature availability:")
    print(features[feature_columns].notna().mean().round(3))


    unique_motion = features[
        ["obs_idx", "gps_speed_mps"]
    ].drop_duplicates("obs_idx")

    print("\nGPS speed statistics (m/s):")
    print(unique_motion["gps_speed_mps"].describe())

    print("\nObservations with GPS speed above 40 m/s:")
    print(
        unique_motion.loc[
            unique_motion["gps_speed_mps"] > 40
        ].head(20).to_string(index=False)
    )

    print("\nHeading difference statistics:")
    print(features["heading_diff_deg"].describe())

    print("\nHeading difference by label:")
    print(
        features.groupby("label")["heading_diff_deg"]
        .describe()
    )


    print("\nFeature sample:")
    print(features.head(10).to_string(index=False))

    print("\nLabel distribution:")
    print(features["label"].value_counts(dropna=False))

    return features, summary


if __name__ == "__main__":
    train_trip_ids = get_trip_ids("train")[:5]
    student_test_trip_ids = get_trip_ids("student_test")[:3]

    for trip_id in train_trip_ids:
        test_feature_pipeline(trip_id, "train")

    for trip_id in student_test_trip_ids:
        #test_feature_pipeline(trip_id, "student_test")
        test_feature_pipeline("00000009", "student_test")
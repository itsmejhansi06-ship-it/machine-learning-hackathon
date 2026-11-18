
import pandas as pd
from pathlib import Path

base = Path("data/processed")

for filename in [
    "train_features.parquet",
    "student_test_features.parquet"
]:
    path = base / filename
    df = pd.read_parquet(path)

    print(f"\n{filename}")
    print("Shape:", df.shape)
    print("Unique trips:", df["trip"].nunique())
    print("Columns:", df.columns.tolist())
    print("Missing values:")
    print(df.isna().sum())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    assert "label" in df.columns
    assert "reference_arc_id" not in df.columns
    assert "alignment_distance" not in df.columns
    assert "route_position" not in df.columns

print("\nMiss-rate summary:")
print(
    pd.read_csv(
        base / "candidate_miss_rates.csv"
    ).to_string(index=False)
)

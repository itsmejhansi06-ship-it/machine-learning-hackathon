
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "train_features.parquet"
OUTPUT_DIR = ROOT / "results" / "feature_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"

# Columns that identify records, not predictive features
EXCLUDE = {
    "label",
    "trip",
    "trip_id",
    "split",
    "obs_idx",
    "t",
    "arc_id",
    "reference_arc_id",
    "alignment_distance",
    "route_position",
    "projection_epsg",
}

print("Loading training dataset...")
df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

# Numeric predictive features only
feature_cols = [
    c for c in df.select_dtypes(include=np.number).columns
    if c not in EXCLUDE
]

# Remove constant and entirely missing features
feature_cols = [
    c for c in feature_cols
    if df[c].notna().any() and df[c].nunique(dropna=True) > 1
]

if not feature_cols:
    raise ValueError("No usable numeric feature columns found.")

df = df.replace([np.inf, -np.inf], np.nan)

# Fill missing values with medians
df[feature_cols] = df[feature_cols].fillna(
    df[feature_cols].median()
)

df = df.dropna(subset=[TARGET])

print("\nFeatures being evaluated:")
for col in feature_cols:
    print(" -", col)

print("\nClass distribution:")
print(df[TARGET].value_counts())

positive = df[df[TARGET] == 1]
negative = df[df[TARGET] == 0]

# --------------------------------------------------
# 1. FEATURE DISTRIBUTIONS
# --------------------------------------------------

print("\nGenerating feature distributions...")

selected = [
    c for c in [
        "distance",
        "cross_track_distance",
        "heading_diff_deg",
        "speed",
        "arc_bearing",
        "projection_fraction",
        "along_arc_distance",
        "arc_length",
    ]
    if c in feature_cols
]

if not selected:
    selected = feature_cols[:8]

fig, axes = plt.subplots(
    len(selected), 1,
    figsize=(10, 3.2 * len(selected))
)

if len(selected) == 1:
    axes = [axes]

for ax, col in zip(axes, selected):
    # Sample for readable, faster plots
    p = positive[col].sample(
        min(10000, len(positive)), random_state=42
    )
    n = negative[col].sample(
        min(10000, len(negative)), random_state=42
    )

    sns.histplot(
        p, ax=ax, color="green",
        label="Positive (correct)", stat="density",
        element="step", fill=False
    )
    sns.histplot(
        n, ax=ax, color="red",
        label="Negative (incorrect)", stat="density",
        element="step", fill=False
    )

    ax.set_title(f"{col}: Positive vs Negative")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "01_feature_distributions.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 2. FEATURE CORRELATION HEATMAP
# --------------------------------------------------

print("Generating correlation heatmap...")

corr = df[feature_cols].corr()

plt.figure(figsize=(
    max(10, len(feature_cols) * 0.9),
    max(8, len(feature_cols) * 0.7)
))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_correlation_heatmap.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 3. FEATURE IMPORTANCE
# --------------------------------------------------

print("Training a diagnostic Random Forest...")

# Cap rows for quick exploratory analysis
sample = df.sample(
    min(100000, len(df)),
    random_state=42
)

X = sample[feature_cols]
y = sample[TARGET].astype(int)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, max(6, len(feature_cols) * 0.45)))

sns.barplot(
    data=importance,
    x="importance",
    y="feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "03_feature_importance.png",
    dpi=200
)
plt.close()

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

# --------------------------------------------------
# 4. POSITIVE VS NEGATIVE BOXPLOTS
# --------------------------------------------------

print("Generating boxplots...")

plot_df = df.sample(
    min(30000, len(df)),
    random_state=42
)

fig, axes = plt.subplots(
    len(selected), 1,
    figsize=(9, 3 * len(selected))
)

if len(selected) == 1:
    axes = [axes]

for ax, col in zip(axes, selected):
    sns.boxplot(
        data=plot_df,
        x=TARGET,
        y=col,
        ax=ax
    )
    ax.set_title(f"{col} by Candidate Class")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_positive_negative_boxplots.png",
    dpi=200
)
plt.close()

# --------------------------------------------------
# 5. FEATURE RELATIONSHIP SCATTER PLOTS
# --------------------------------------------------

print("Generating feature relationship plots...")

pairs = [
    ("distance", "heading_diff_deg"),
    ("speed", "heading_diff_deg"),
    ("distance", "projection_fraction"),
    ("cross_track_distance", "along_arc_distance"),
]

pairs = [
    (a, b) for a, b in pairs
    if a in feature_cols and b in feature_cols and a != b
]

if pairs:
    fig, axes = plt.subplots(
        len(pairs), 1,
        figsize=(9, 5 * len(pairs))
    )

    if len(pairs) == 1:
        axes = [axes]

    for ax, (x, y_col) in zip(axes, pairs):
        sns.scatterplot(
            data=plot_df,
            x=x,
            y=y_col,
            hue=TARGET,
            alpha=0.35,
            s=15,
            ax=ax,
            palette={0: "red", 1: "green"}
        )
        ax.set_title(f"{x} vs {y_col}")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "05_feature_relationships.png",
        dpi=200
    )
    plt.close()

# --------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------

summary = pd.DataFrame({
    "feature": feature_cols,
    "positive_mean": [
        positive[c].mean() for c in feature_cols
    ],
    "negative_mean": [
        negative[c].mean() for c in feature_cols
    ],
    "positive_median": [
        positive[c].median() for c in feature_cols
    ],
    "negative_median": [
        negative[c].median() for c in feature_cols
    ],
    "correlation_with_label": [
        df[c].corr(df[TARGET]) for c in feature_cols
    ]
})

summary.to_csv(
    OUTPUT_DIR / "feature_evaluation_summary.csv",
    index=False
)

print("\n====================================")
print("FEATURE EVALUATION COMPLETED")
print("====================================")
print("Output directory:", OUTPUT_DIR)
print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

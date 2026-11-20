
# ==============================================================================
# MODULE 11
# APPLICABILITY DOMAIN-RELATED ANALYSIS
# RANDOM FOREST QSAR MODEL
# Ensemble Disagreement Framework
# ==============================================================================

# ==============================================================================
# 1. IMPORT LIBRARIES
# ==============================================================================

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# ==============================================================================
# 2. PROJECT PATHS
# ==============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_DIR / "outputs" / "model"

MODEL_FILE = MODEL_DIR / "rf_qsar_model.pkl"
SCALER_FILE = MODEL_DIR / "rf_scaler.pkl"
FEATURE_NAMES_FILE = MODEL_DIR / "rf_feature_names.pkl"

X_TEST_FILE = MODEL_DIR / "rf_X_test_final.csv"
Y_TEST_FILE = MODEL_DIR / "rf_y_test_final.csv"

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "applicability_domain"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 3. LOAD TRAINED MODEL
# ==============================================================================

print("\nLoading trained RF model...")

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_NAMES_FILE,
    X_TEST_FILE,
    Y_TEST_FILE
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

rf_model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_names = joblib.load(FEATURE_NAMES_FILE)

print("\nRF model loaded.")
print("Number of model features:", len(feature_names))


# ==============================================================================
# 4. LOAD SAVED TEST SET
# ==============================================================================

X_test = pd.read_csv(X_TEST_FILE)
y_test = pd.read_csv(Y_TEST_FILE)

if isinstance(y_test, pd.DataFrame):
    if y_test.shape[1] != 1:
        raise ValueError(
            "The saved test target file must contain exactly "
            "one column."
        )
    y_test = y_test.iloc[:, 0]

y_test = pd.to_numeric(
    y_test,
    errors="coerce"
)

if y_test.isna().any():
    raise ValueError(
        "The saved test labels contain missing or "
        "non-numeric values."
    )

if not set(y_test.unique()).issubset({0, 1}):
    raise ValueError(
        "Test labels must contain only 0 (INACTIVE) "
        "and 1 (ACTIVE)."
    )

y_test = y_test.astype(int).reset_index(drop=True)

print("\nTest set loaded.")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==============================================================================
# 5. MATCH TEST FEATURES WITH TRAINING FEATURES
# ==============================================================================

print("\nMatching test features with training features...")

missing_features = [
    feature
    for feature in feature_names
    if feature not in X_test.columns
]

if missing_features:
    raise ValueError(
        "The test set is missing required model features:\n"
        f"{missing_features}"
    )

X_test = X_test.loc[:, feature_names].copy()

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

if X_test.isna().any().any():
    missing_counts = X_test.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    raise ValueError(
        "Missing or non-numeric test descriptors detected:\n"
        f"{missing_counts.to_string()}"
    )

print("Matched X_test shape:", X_test.shape)


# ==============================================================================
# 6. CONVERT TO FLOAT32
# ==============================================================================

FLOAT32_MAX = np.finfo(np.float32).max

X_test = X_test.clip(
    lower=-FLOAT32_MAX,
    upper=FLOAT32_MAX
)

X_test = X_test.astype(np.float32)

if len(X_test) != len(y_test):
    raise ValueError(
        "Test feature and target files contain different "
        "numbers of observations."
    )


# ==============================================================================
# 7. SCALE TEST SET
# ==============================================================================

print("\nScaling test features...")

X_test_scaled = scaler.transform(X_test)

X_test_scaled = np.asarray(
    X_test_scaled,
    dtype=np.float32
)

print(
    "Scaled test matrix shape:",
    X_test_scaled.shape
)


# ==============================================================================
# 8. EXTRACT ACTIVE-CLASS PROBABILITIES FROM EVERY TREE
# ==============================================================================

print("\nCalculating ensemble disagreement...")

if 1 not in rf_model.classes_:
    raise ValueError(
        "The trained RF model does not contain active class 1."
    )

tree_probabilities = []

for tree in rf_model.estimators_:

    tree_classes = list(tree.classes_)

    if 1 in tree_classes:

        active_index = tree_classes.index(1)

        probabilities = tree.predict_proba(
            X_test_scaled
        )[:, active_index]

    else:
        # A tree that contains no active-class samples
        # assigns zero probability to the active class.
        probabilities = np.zeros(
            X_test_scaled.shape[0]
        )

    tree_probabilities.append(probabilities)


tree_probabilities = np.asarray(
    tree_probabilities
)

print(
    "Tree probability matrix shape:",
    tree_probabilities.shape
)


# ==============================================================================
# 9. CALCULATE ENSEMBLE DISAGREEMENT
# ==============================================================================

# Variance of active-class probabilities across RF trees.
#
# Higher variance indicates greater disagreement between
# individual trees.

ensemble_variance = np.var(
    tree_probabilities,
    axis=0
)


# ==============================================================================
# 10. FINAL MODEL PREDICTIONS
# ==============================================================================

active_class_index = list(
    rf_model.classes_
).index(1)

final_probs = rf_model.predict_proba(
    X_test_scaled
)[:, active_class_index]

final_preds = rf_model.predict(
    X_test_scaled
)


# ==============================================================================
# 11. OVERALL TEST SET ACCURACY
# ==============================================================================

global_acc = accuracy_score(
    y_test,
    final_preds
)

print(
    "\nGlobal test set accuracy:",
    f"{global_acc:.4f}"
)


# ==============================================================================
# 12. CREATE AD-RELATED RESULTS DATAFRAME
# ==============================================================================

ad_df = pd.DataFrame({
    "y_true": y_test.to_numpy(),
    "y_pred": final_preds,
    "prob_active": final_probs,
    "disagreement": ensemble_variance,
    "correct": (
        final_preds == y_test.to_numpy()
    ).astype(int)
})


# ==============================================================================
# 13. CALCULATE TRUE-CLASS PREDICTION MARGIN
# ==============================================================================

# Positive margin:
# The model assigns greater probability to the true class.
#
# Negative margin:
# The model assigns greater probability to the incorrect class.

ad_df["true_class_margin"] = np.where(
    ad_df["y_true"] == 1,
    2 * ad_df["prob_active"] - 1,
    1 - 2 * ad_df["prob_active"]
)


# ==============================================================================
# 14. PLOT STYLE
# ==============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


# ==============================================================================
# PLOT 1
# COVERAGE VS ACCURACY
# ==============================================================================

print("\nCreating Coverage vs Accuracy plot...")

# Sort molecules from lowest disagreement
# to highest disagreement.

df_coverage = (
    ad_df
    .sort_values(
        by="disagreement",
        ascending=True
    )
    .reset_index(drop=True)
)

coverages = np.linspace(
    10,
    100,
    10
)

accuracies = []

for cov in coverages:

    n_samples = int(
        len(df_coverage) * cov / 100
    )

    n_samples = max(n_samples, 1)

    sub_df = df_coverage.iloc[:n_samples]

    acc = accuracy_score(
        sub_df["y_true"],
        sub_df["y_pred"]
    )

    accuracies.append(acc)


plt.figure(figsize=(5.5, 4.5))

plt.plot(
    coverages,
    accuracies,
    marker="o",
    color="navy",
    linewidth=2
)

plt.axhline(
    global_acc,
    color="crimson",
    linestyle="--",
    label=f"Global Accuracy = {global_acc:.2%}"
)

plt.xlabel("Dataset Coverage (%)")
plt.ylabel("Accuracy")
plt.title("Coverage vs Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ad_coverage_vs_accuracy.png"
)

plt.close()


# ==============================================================================
# PLOT 2
# BIN-BASED AVERAGING
# ==============================================================================

print("\nCreating Bin-Based Averaging plot...")

# Rank values first to handle tied disagreement values
# while forming four approximately equal-sized groups.

ad_df["bin"] = pd.qcut(
    ad_df["disagreement"].rank(method="first"),
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"]
)

bba_summary = (
    ad_df
    .groupby(
        "bin",
        observed=False
    )
    .agg(
        bin_accuracy=("correct", "mean"),
        count=("correct", "count")
    )
    .reset_index()
)

plt.figure(figsize=(5.5, 4.5))

bars = plt.bar(
    bba_summary["bin"].astype(str),
    bba_summary["bin_accuracy"],
    color="#34495e",
    edgecolor="black"
)

for bar, count in zip(
    bars,
    bba_summary["count"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"n={count}",
        ha="center"
    )

plt.ylim(0, 1.05)
plt.xlabel("Tree Disagreement Quartiles")
plt.ylabel("Accuracy")
plt.title("Bin-Based Averaging")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ad_bin_based_averaging.png"
)

plt.close()


# ==============================================================================
# PLOT 3
# DISAGREEMENT VS TRUE-CLASS PREDICTION MARGIN
# ==============================================================================

print(
    "\nCreating Disagreement vs Prediction Margin plot..."
)

plt.figure(figsize=(6, 5))

correct_pts = ad_df[
    ad_df["correct"] == 1
]

incorrect_pts = ad_df[
    ad_df["correct"] == 0
]

plt.scatter(
    correct_pts["disagreement"],
    correct_pts["true_class_margin"],
    color="#2ecc71",
    edgecolor="black",
    alpha=0.7,
    s=35,
    label="Correct"
)

plt.scatter(
    incorrect_pts["disagreement"],
    incorrect_pts["true_class_margin"],
    color="#e74c3c",
    edgecolor="black",
    alpha=0.7,
    s=35,
    label="Misclassified"
)

# Empirical 75th-percentile disagreement threshold.

warning_line = np.percentile(
    ad_df["disagreement"],
    75
)

plt.axvline(
    warning_line,
    color="gray",
    linestyle=":",
    linewidth=1.5,
    label="75th percentile disagreement"
)

plt.axhline(
    0,
    color="black",
    linewidth=1
)

plt.xlabel("Ensemble Disagreement (Probability Variance)")
plt.ylabel("True-Class Prediction Margin")
plt.title("Disagreement vs Prediction Margin")

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ad_disagreement_margin_plot.png"
)

plt.close()


# ==============================================================================
# 15. SAVE AD-RELATED RESULTS
# ==============================================================================

ad_df.to_csv(
    OUTPUT_DIR / "rf_applicability_domain_results.csv",
    index=False
)


# ==============================================================================
# 16. SAVE AD SUMMARY
# ==============================================================================

ad_summary = pd.DataFrame({
    "Metric": [
        "Global_Test_Accuracy",
        "Mean_Disagreement",
        "Median_Disagreement",
        "75th_Percentile_Disagreement",
        "Maximum_Disagreement"
    ],
    "Value": [
        global_acc,
        ad_df["disagreement"].mean(),
        ad_df["disagreement"].median(),
        warning_line,
        ad_df["disagreement"].max()
    ]
})

ad_summary.to_csv(
    OUTPUT_DIR / "rf_applicability_domain_summary.csv",
    index=False
)


# ==============================================================================
# 17. CONSOLE SUMMARY
# ==============================================================================

print("\n" + "=" * 55)
print("ENSEMBLE DISAGREEMENT ANALYSIS SUMMARY")
print("=" * 55)

print(f"\nGlobal Test Accuracy : {global_acc:.4f}")

print(
    "Mean Disagreement    : "
    f"{ad_df['disagreement'].mean():.6f}"
)

print(
    "Median Disagreement  : "
    f"{ad_df['disagreement'].median():.6f}"
)

print(
    "75th Percentile      : "
    f"{warning_line:.6f}"
)

print(
    "Maximum Disagreement : "
    f"{ad_df['disagreement'].max():.6f}"
)

print("\nCoverage vs Accuracy:")

for cov, acc in zip(coverages, accuracies):

    print(
        f"   Top {int(cov)}% most confident subset "
        f"-> Accuracy: {acc:.4f}"
    )

print("\nBin-Based Quartile Accuracies:")

for _, row in bba_summary.iterrows():

    print(
        f"   {row['bin']} "
        f"-> Accuracy: {row['bin_accuracy']:.4f} "
        f"(n={row['count']})"
    )

print("\nEmpirical disagreement warning threshold:")
print(f"   {warning_line:.6f}")

print("\nFiles saved:")
print("rf_applicability_domain_results.csv")
print("rf_applicability_domain_summary.csv")
print("ad_coverage_vs_accuracy.png")
print("ad_bin_based_averaging.png")
print("ad_disagreement_margin_plot.png")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nModule 11 completed.")


# ==============================================================================
# END MODULE 11
# ==============================================================================

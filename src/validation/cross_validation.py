
# ============================================================
# MODULE 10
# 5-FOLD CROSS VALIDATION
# RANDOM FOREST QSAR MODEL
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    roc_curve
)

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection"
    / "rf_final_qsar_dataset.csv"
)

MODEL_DIR = PROJECT_DIR / "outputs" / "model"

FEATURE_NAMES_FILE = (
    MODEL_DIR / "rf_feature_names.pkl"
)

SELECTED_FEATURES_FILE = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection"
    / "selected_descriptor_names.txt"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "cross_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading final QSAR dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"QSAR dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded.")
print("Dataset shape:", df.shape)


# ============================================================
# 4. LOAD FINAL RF FEATURE NAMES
# ============================================================

if not FEATURE_NAMES_FILE.exists():
    raise FileNotFoundError(
        f"RF feature list not found: {FEATURE_NAMES_FILE}"
    )

feature_names = joblib.load(FEATURE_NAMES_FILE)

print(
    "\nFeatures expected by RF model:",
    len(feature_names)
)


# ============================================================
# 5. VERIFY FEATURE LIST
# ============================================================

if not SELECTED_FEATURES_FILE.exists():
    raise FileNotFoundError(
        "Selected descriptor list from Module 6 was not found: "
        f"{SELECTED_FEATURES_FILE}"
    )

selected_descriptors = (
    pd.read_csv(
        SELECTED_FEATURES_FILE,
        header=None
    )[0]
    .astype(str)
    .tolist()
)

if feature_names != selected_descriptors:
    raise ValueError(
        "The Module 7 feature list does not match the "
        "selected descriptor list from Module 6. "
        "Check feature names and their order."
    )


# ============================================================
# 6. DEFINE TARGET
# ============================================================

if "Activity_Class" not in df.columns:
    raise ValueError(
        "Activity_Class column not found in the dataset."
    )

y = pd.to_numeric(
    df["Activity_Class"],
    errors="coerce"
)

if y.isna().any():
    raise ValueError(
        "Activity_Class contains missing or non-numeric values."
    )

if not set(y.unique()).issubset({0, 1}):
    raise ValueError(
        "Activity_Class must contain only binary labels: "
        "0 (INACTIVE) and 1 (ACTIVE)."
    )

y = y.astype(int)


# ============================================================
# 7. EXTRACT EXACT MODEL FEATURES
# ============================================================
#
# Use only the descriptors saved with the trained RF model.
# This prevents numeric metadata from entering the feature
# matrix accidentally.
#
# ============================================================

missing_features = [
    feature
    for feature in feature_names
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "The following model features are missing from "
        f"the QSAR dataset: {missing_features}"
    )

X = df.loc[:, feature_names].copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

if X.isna().any().any():
    missing_counts = X.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    raise ValueError(
        "Missing or non-numeric descriptor values detected:\n"
        f"{missing_counts.to_string()}"
    )


# ============================================================
# 8. CONVERT TO FLOAT32
# ============================================================

FLOAT32_MAX = np.finfo(np.float32).max

X = X.clip(
    lower=-FLOAT32_MAX,
    upper=FLOAT32_MAX
)

X = X.astype(np.float32)


# ============================================================
# 9. FINAL DATA CHECK
# ============================================================

print("\n===================================")
print("CROSS-VALIDATION FEATURE CHECK")
print("===================================")

print("Feature shape:", X.shape)
print("Expected features:", len(feature_names))
print("Remaining NaN:", X.isna().sum().sum())
print("Remaining Inf:", np.isinf(X.to_numpy()).sum())
print("Maximum value:", np.max(X.to_numpy()))
print("Minimum value:", np.min(X.to_numpy()))

if X.shape[1] != len(feature_names):
    raise ValueError(
        "Feature count does not match rf_feature_names.pkl."
    )

if len(X) != len(y):
    raise ValueError(
        "Feature matrix and target contain different numbers "
        "of observations."
    )


# ============================================================
# 10. CHECK CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nClass proportions:")
print(y.value_counts(normalize=True).sort_index())

class_counts = y.value_counts()

if len(class_counts) != 2:
    raise ValueError(
        "Both classes (0 and 1) must be present for "
        "binary classification."
    )

if class_counts.min() < 5:
    raise ValueError(
        "At least five observations per class are required "
        "for stratified 5-fold cross-validation."
    )


# ============================================================
# 11. CONVERT TO ARRAYS
# ============================================================

X_arr = X.to_numpy()
y_arr = y.to_numpy()


# ============================================================
# 12. RANDOM FOREST CONFIGURATION
# ============================================================
#
# Same RF configuration as Module 7.
#
# ============================================================

rf_parameters = {
    "n_estimators": 1000,
    "max_depth": 15,
    "min_samples_split": 8,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1
}


# ============================================================
# 13. STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 14. RESULT STORAGE
# ============================================================

fold_roc_aucs = []
fold_bal_accs = []

tprs = []

mean_fpr = np.linspace(
    0,
    1,
    100
)


# ============================================================
# 15. CROSS-VALIDATION LOOP
# ============================================================

print("\n===================================")
print("STARTING 5-FOLD CROSS VALIDATION")
print("===================================")

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_arr, y_arr),
    start=1
):

    print(f"\nRunning Fold {fold}...")

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    X_train = X_arr[train_idx]
    X_test = X_arr[test_idx]

    y_train = y_arr[train_idx]
    y_test = y_arr[test_idx]

    # --------------------------------------------------------
    # Train RF
    # --------------------------------------------------------

    rf_model = RandomForestClassifier(
        **rf_parameters
    )

    rf_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = rf_model.predict(X_test)

    active_class_index = list(
        rf_model.classes_
    ).index(1)

    y_proba = rf_model.predict_proba(
        X_test
    )[:, active_class_index]

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_proba,
        pos_label=1
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    # --------------------------------------------------------
    # Balanced accuracy
    # --------------------------------------------------------

    bal_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    fold_roc_aucs.append(roc_auc)
    fold_bal_accs.append(bal_acc)

    # --------------------------------------------------------
    # Interpolate ROC
    # --------------------------------------------------------

    interp_tpr = np.interp(
        mean_fpr,
        fpr,
        tpr
    )

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

    # --------------------------------------------------------
    # Print fold result
    # --------------------------------------------------------

    print(f"Fold {fold}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")


# ============================================================
# 16. CREATE CV RESULTS TABLE
# ============================================================

cv_results = pd.DataFrame({
    "Fold": range(1, 6),
    "ROC_AUC": fold_roc_aucs,
    "Balanced_Accuracy": fold_bal_accs
})


# ============================================================
# 17. CALCULATE SUMMARY STATISTICS
# ============================================================

mean_roc_auc = cv_results["ROC_AUC"].mean()
std_roc_auc = cv_results["ROC_AUC"].std()

mean_bal_acc = cv_results["Balanced_Accuracy"].mean()
std_bal_acc = cv_results["Balanced_Accuracy"].std()


# ============================================================
# 18. SAVE CV RESULTS
# ============================================================

cv_results.to_csv(
    OUTPUT_DIR / "rf_cross_validation_results.csv",
    index=False
)

summary_df = pd.DataFrame({
    "Metric": [
        "ROC_AUC",
        "Balanced_Accuracy"
    ],
    "Mean": [
        mean_roc_auc,
        mean_bal_acc
    ],
    "Std": [
        std_roc_auc,
        std_bal_acc
    ]
})

summary_df.to_csv(
    OUTPUT_DIR / "rf_cross_validation_summary.csv",
    index=False
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print("\n===================================")
print("5-FOLD CROSS VALIDATION RESULTS")
print("===================================")

print("\nFold-wise results:")
print(cv_results.to_string(index=False))

print("\nMean ROC-AUC:")
print(f"{mean_roc_auc:.4f} ± {std_roc_auc:.4f}")

print("\nMean Balanced Accuracy:")
print(f"{mean_bal_acc:.4f} ± {std_bal_acc:.4f}")


# ============================================================
# 20. PREPARE MEAN ROC CURVE
# ============================================================

mean_tpr = np.mean(
    tprs,
    axis=0
)

mean_tpr[-1] = 1.0

mean_auc = auc(
    mean_fpr,
    mean_tpr
)

std_tpr = np.std(
    tprs,
    axis=0
)

tprs_upper = np.minimum(
    mean_tpr + std_tpr,
    1
)

tprs_lower = np.maximum(
    mean_tpr - std_tpr,
    0
)


# ============================================================
# 21. PUBLICATION-STYLE ROC PLOT
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

fig, ax = plt.subplots(
    figsize=(6, 5)
)


# ============================================================
# 22. PLOT EACH FOLD
# ============================================================

for tpr in tprs:
    ax.plot(
        mean_fpr,
        tpr,
        lw=1,
        alpha=0.30,
        linestyle="--",
        color="dimgray"
    )


# ============================================================
# 23. CHANCE LINE
# ============================================================

ax.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    color="black",
    lw=1.2,
    label="Chance (AUC = 0.50)"
)


# ============================================================
# 24. MEAN ROC
# ============================================================

ax.plot(
    mean_fpr,
    mean_tpr,
    color="navy",
    lw=2.0,
    alpha=0.9,
    label=(
        f"Mean ROC "
        f"(AUC = {mean_auc:.2f} "
        f"$\\pm$ {std_roc_auc:.2f})"
    )
)


# ============================================================
# 25. STANDARD DEVIATION BAND
# ============================================================

ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="blue",
    alpha=0.08,
    label=r"$\pm$ 1 std. dev."
)


# ============================================================
# 26. AXIS AND TITLE
# ============================================================

ax.set(
    xlim=[-0.02, 1.02],
    ylim=[-0.02, 1.02],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Random Forest 5-Fold Cross Validation"
)


# ============================================================
# 27. GRID AND LEGEND
# ============================================================

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend(
    loc="lower right",
    fontsize=8.5,
    frameon=True,
    facecolor="white",
    edgecolor="none"
)


# ============================================================
# 28. SAVE FIGURES
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rf_cross_validation_roc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "rf_cross_validation_roc.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n===================================")
print("MODULE 10 COMPLETED")
print("===================================")

print("\nFiles saved:")
print("rf_cross_validation_results.csv")
print("rf_cross_validation_summary.csv")
print("rf_cross_validation_roc.png")
print("rf_cross_validation_roc.pdf")

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# END OF MODULE 10
# ============================================================

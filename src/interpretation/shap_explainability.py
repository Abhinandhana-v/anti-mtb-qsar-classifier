
# ============================================================
# MODULE 9
# SHAP ANALYSIS
# RANDOM FOREST QSAR MODEL
# ============================================================

import time
import joblib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection"
    / "rf_final_qsar_dataset.csv"
)

MODEL_DIR = PROJECT_DIR / "outputs" / "model"

MODEL_FILE = MODEL_DIR / "rf_qsar_model.pkl"
FEATURE_NAMES_FILE = MODEL_DIR / "rf_feature_names.pkl"

SELECTED_FEATURES_FILE = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection"
    / "selected_descriptor_names.txt"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "interpretation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TIMER
# ============================================================

start_time = time.time()


# ============================================================
# PUBLICATION PLOT STYLING
# ============================================================

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans"
]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0


# ============================================================
# 1. LOAD TRAINED RF MODEL
# ============================================================

print("\nLoading trained RF model...")

for file_path in [MODEL_FILE, FEATURE_NAMES_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required model file not found: {file_path}"
        )

rf_model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_NAMES_FILE)

print("RF model loaded.")
print("Number of model features:", len(feature_names))


# ============================================================
# 2. VERIFY SELECTED DESCRIPTOR LIST
# ============================================================

if not SELECTED_FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Selected descriptor list not found: "
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
        "The model feature list does not match the "
        "selected descriptor list from Module 6. "
        "Check the feature order and Module 7 outputs."
    )


# ============================================================
# 3. LOAD FINAL QSAR DATASET
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
# 4. EXTRACT EXACT MODEL FEATURES
# ============================================================
#
# Use only the features saved with the trained RF model.
# This prevents metadata and activity-related columns from
# accidentally entering the SHAP feature matrix.
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

# Ensure all model features are numeric.
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with NaN.
X = X.replace([np.inf, -np.inf], np.nan)

# Do not silently replace missing descriptor values with zero.
if X.isna().any().any():
    missing_counts = X.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    raise ValueError(
        "Missing or non-numeric values detected in the "
        "SHAP feature matrix:\n"
        f"{missing_counts.to_string()}"
    )

print("\nSHAP feature matrix prepared.")
print("Feature matrix shape:", X.shape)


# ============================================================
# 5. SAMPLE DATA FOR SHAP
# ============================================================

sample_size = min(1000, len(X))

if sample_size == 0:
    raise ValueError("The QSAR dataset contains no molecules.")

X_sample = X.sample(
    n=sample_size,
    random_state=42
)

print("\nSHAP sample size:", X_sample.shape)


# ============================================================
# 6. IDENTIFY ACTIVE CLASS
# ============================================================

model_classes = list(rf_model.classes_)

if 1 not in model_classes:
    raise ValueError(
        "The trained RF model does not contain class 1 "
        "(ACTIVE)."
    )

active_class_index = model_classes.index(1)


# ============================================================
# 7. CREATE SHAP TREE EXPLAINER
# ============================================================

print("\n===================================")
print("SHAP ANALYSIS")
print("===================================")

print("\nCreating TreeExplainer...")

t0 = time.time()

explainer = shap.TreeExplainer(rf_model)

print(
    "TreeExplainer time:",
    f"{time.time() - t0:.2f} sec"
)


# ============================================================
# 8. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

t1 = time.time()

shap_values_raw = explainer.shap_values(X_sample)

print(
    "SHAP calculation time:",
    f"{time.time() - t1:.2f} sec"
)


# ============================================================
# 9. EXTRACT ACTIVE-CLASS SHAP VALUES
# ============================================================

if isinstance(shap_values_raw, list):

    # Older SHAP versions
    selected_shap_values = (
        shap_values_raw[active_class_index]
    )

elif isinstance(shap_values_raw, np.ndarray):

    if shap_values_raw.ndim == 3:

        # Newer SHAP versions:
        # samples x features x classes
        selected_shap_values = (
            shap_values_raw[:, :, active_class_index]
        )

    elif shap_values_raw.ndim == 2:

        # SHAP output already corresponds to one class
        selected_shap_values = shap_values_raw

    else:
        raise ValueError(
            "Unexpected SHAP array dimensions: "
            f"{shap_values_raw.shape}"
        )

else:
    raise TypeError(
        "Unexpected SHAP output type: "
        f"{type(shap_values_raw)}"
    )


selected_shap_values = np.asarray(selected_shap_values)

expected_shape = (
    X_sample.shape[0],
    X_sample.shape[1]
)

if selected_shap_values.shape != expected_shape:
    raise ValueError(
        "SHAP values do not match the feature matrix. "
        f"Expected {expected_shape}, "
        f"received {selected_shap_values.shape}."
    )

print("\nActive-class SHAP value shape:")
print(selected_shap_values.shape)


# ============================================================
# 10. CREATE SHAP EXPLANATION OBJECT
# ============================================================

expected_value = explainer.expected_value

if isinstance(expected_value, (list, np.ndarray)):
    base_value = np.asarray(expected_value)[active_class_index]
else:
    base_value = expected_value

base_values = np.full(
    shape=X_sample.shape[0],
    fill_value=float(base_value)
)

shap_exp = shap.Explanation(
    values=selected_shap_values,
    base_values=base_values,
    data=X_sample.values,
    feature_names=X_sample.columns.tolist()
)


# ============================================================
# 11. SHAP SUMMARY PLOT: BEESWARM
# ============================================================

print("\nCreating SHAP beeswarm plot...")

t2 = time.time()

shap.summary_plot(
    shap_exp,
    X_sample,
    max_display=20,
    plot_type="dot",
    show=False
)

plt.xlabel(
    "SHAP Value (Impact on Active-Class Prediction)",
    fontsize=11,
    fontweight="bold"
)

plt.title(
    "Feature Impact on QSAR Model Predictions",
    fontsize=12,
    fontweight="bold",
    pad=15
)

plt.tick_params(labelsize=10)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rf_shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Beeswarm plot time:",
    f"{time.time() - t2:.2f} sec"
)


# ============================================================
# 12. SHAP BAR PLOT: GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\nCreating SHAP bar plot...")

t3 = time.time()

shap.summary_plot(
    shap_exp,
    X_sample,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.xlabel(
    "Mean |SHAP Value| (Average Predictor Impact)",
    fontsize=11,
    fontweight="bold"
)

plt.title(
    "Global SHAP Feature Importance",
    fontsize=12,
    fontweight="bold",
    pad=15
)

plt.tick_params(labelsize=10)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rf_shap_bar_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Bar plot time:",
    f"{time.time() - t3:.2f} sec"
)


# ============================================================
# 13. CALCULATE GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(
    selected_shap_values
).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean_Absolute_SHAP": mean_abs_shap
})

shap_importance_df = (
    shap_importance_df
    .sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 14. SAVE SHAP IMPORTANCE TABLE
# ============================================================

shap_importance_df.to_csv(
    OUTPUT_DIR / "rf_shap_feature_importance.csv",
    index=False
)


# ============================================================
# 15. DISPLAY TOP FEATURES
# ============================================================

print("\n===================================")
print("TOP 20 SHAP FEATURES")
print("===================================")

print(
    shap_importance_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 16. SAVE TOP 20 FEATURES
# ============================================================

shap_importance_df.head(20).to_csv(
    OUTPUT_DIR / "rf_shap_top20_features.csv",
    index=False
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

elapsed_time = time.time() - start_time

print("\n===================================")
print("MODULE 9 COMPLETED")
print("===================================")

print("\nRF features explained:", X.shape[1])
print("Molecules used for SHAP:", X_sample.shape[0])

print("\nFiles generated:")

print("rf_shap_summary_plot.png")
print("rf_shap_bar_plot.png")
print("rf_shap_feature_importance.csv")
print("rf_shap_top20_features.csv")

print(
    "\nTotal execution time:",
    f"{elapsed_time:.2f} seconds"
)

# ============================================================
# END OF MODULE 9
# ============================================================


# ============================================================
# MODULE 13
# GENERIC RANDOM FOREST VIRTUAL SCREENING
# SHAP-PRIORITIZED OUTPUT
# ============================================================

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS AND LIBRARY CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Must match the library name used in Module 12.
LIBRARY_NAME = "Screening Library"

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "virtual_screening"
    / LIBRARY_NAME
)

INPUT_FILE = (
    OUTPUT_DIR
    / f"{LIBRARY_NAME}_descriptor_dataset.csv"
)

MODEL_DIR = PROJECT_DIR / "outputs" / "model"

MODEL_FILE = MODEL_DIR / "rf_qsar_model.pkl"
SCALER_FILE = MODEL_DIR / "rf_scaler.pkl"
FEATURE_FILE = MODEL_DIR / "rf_feature_names.pkl"

ALL_OUTPUT_FILE = (
    OUTPUT_DIR
    / f"{LIBRARY_NAME}_Classifier_All.csv"
)

TOP_OUTPUT_FILE = (
    OUTPUT_DIR
    / f"{LIBRARY_NAME}_Classifier_Top500.csv"
)


# ============================================================
# 2. SHAP-IMPORTANT FEATURES
# ============================================================

SHAP_TOP_FEATURES = [
    "PEOE_VSA9",
    "PEOE_VSA6",
    "Kappa3",
    "EState_VSA5",
    "PEOE_VSA12",
    "VSA_EState8",
    "SMR_VSA7",
    "AvgIpc",
    "VSA_EState3",
    "MinEStateIndex",
    "PEOE_VSA14",
    "SMR_VSA10",
    "VSA_EState2",
    "VSA_EState7",
    "SPS",
    "PEOE_VSA2",
    "SMR_VSA9",
    "SlogP_VSA3",
    "EState_VSA7",
    "VSA_EState10"
]

MANDATORY_SHAP_FEATURES = [
    "PEOE_VSA9",
    "PEOE_VSA6",
    "Kappa3",
    "EState_VSA5",
    "PEOE_VSA12"
]


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

print("=" * 60)
print("MODULE 13: RANDOM FOREST VIRTUAL SCREENING")
print("=" * 60)

required_files = [
    INPUT_FILE,
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_FILE
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

print("\nLibrary:", LIBRARY_NAME)
print("Input:", INPUT_FILE)
print("Model:", MODEL_FILE)
print("Scaler:", SCALER_FILE)
print("Feature names:", FEATURE_FILE)


# ============================================================
# 4. LOAD SCREENING DATASET
# ============================================================

screen_df = pd.read_csv(INPUT_FILE)

if "SMILES" not in screen_df.columns:
    raise ValueError("SMILES column is missing from screening dataset.")

if screen_df.empty:
    raise ValueError("Screening dataset is empty.")

print("\nScreening dataset:", screen_df.shape)


# ============================================================
# 5. LOAD CLASSIFIER ASSETS
# ============================================================

rf_classifier = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
trained_features = joblib.load(FEATURE_FILE)

trained_features = list(trained_features)

if not trained_features:
    raise ValueError("The trained feature list is empty.")

if len(trained_features) != len(set(trained_features)):
    raise ValueError("Duplicate feature names in trained feature list.")

print("\nClassifier assets loaded.")
print("Number of trained features:", len(trained_features))


# ============================================================
# 6. VERIFY SCREENING FEATURES
# ============================================================

missing_features = [
    feature
    for feature in trained_features
    if feature not in screen_df.columns
]

if missing_features:
    raise ValueError(
        "Required training descriptors are missing from the screening "
        f"dataset:\n{missing_features}\n\n"
        "Regenerate descriptors using Module 12."
    )

# Select descriptors in exactly the training order.
X = screen_df[trained_features].copy()

# Convert to numeric and reject missing/non-finite values.
X = X.apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    bad_features = X.columns[X.isna().any()].tolist()
    raise ValueError(
        f"Missing or non-numeric values in descriptors: {bad_features}"
    )

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite values found in screening descriptors.")

print("Feature alignment: OK")
print("Feature matrix:", X.shape)


# ============================================================
# 7. SCALE FEATURES
# ============================================================

X_scaled = scaler.transform(X)

if not np.isfinite(X_scaled).all():
    raise ValueError("Non-finite values detected after scaling.")

print("Feature scaling completed.")


# ============================================================
# 8. RANDOM FOREST PREDICTION
# ============================================================

prediction = rf_classifier.predict(X_scaled)

model_classes = list(rf_classifier.classes_)

if 1 not in model_classes:
    raise ValueError(
        "The trained classifier does not contain class 1 (ACTIVE)."
    )

active_class_index = model_classes.index(1)

probabilities = rf_classifier.predict_proba(X_scaled)

active_probability = probabilities[:, active_class_index]


# ============================================================
# 9. ADD PREDICTIONS
# ============================================================

screen_df["Predicted_Activity_Class"] = prediction

screen_df["Predicted_Activity"] = np.where(
    prediction == 1,
    "ACTIVE",
    "INACTIVE"
)

screen_df["Active_Probability"] = active_probability


# ============================================================
# 10. SORT AND RANK MOLECULES
# ============================================================

results_df = screen_df.sort_values(
    by="Active_Probability",
    ascending=False
).reset_index(drop=True)

results_df.insert(
    0,
    "Screening_Rank",
    np.arange(1, len(results_df) + 1)
)


# ============================================================
# 11. REORDER OUTPUT COLUMNS
# ============================================================

core_columns = [
    "Screening_Rank",
    "Source_File",
    "Title",
    "SMILES",
    "Predicted_Activity",
    "Predicted_Activity_Class",
    "Active_Probability"
]

core_columns = [
    col for col in core_columns
    if col in results_df.columns
]

available_shap_features = [
    feature for feature in SHAP_TOP_FEATURES
    if feature in results_df.columns
]

remaining_columns = [
    col for col in results_df.columns
    if col not in core_columns
    and col not in available_shap_features
]

results_df = results_df[
    core_columns
    + available_shap_features
    + remaining_columns
]


# ============================================================
# 12. GENERATE TOP 500 RESULTS
# ============================================================

top500_df = results_df.head(500).copy()


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv(ALL_OUTPUT_FILE, index=False)
top500_df.to_csv(TOP_OUTPUT_FILE, index=False)


# ============================================================
# 14. SCREENING SUMMARY
# ============================================================

active_count = (
    results_df["Predicted_Activity"] == "ACTIVE"
).sum()

inactive_count = (
    results_df["Predicted_Activity"] == "INACTIVE"
).sum()

print("\n" + "=" * 60)
print("CLASSIFIER SCREENING COMPLETE")
print("=" * 60)

print("Total molecules:", len(results_df))
print("Predicted ACTIVE:", active_count)
print("Predicted INACTIVE:", inactive_count)

if len(results_df) > 0:
    print(
        "Highest active probability:",
        f"{results_df['Active_Probability'].iloc[0]:.4f}"
    )


# ============================================================
# 15. DISPLAY TOP 10 MOLECULES
# ============================================================

display_columns = [
    "Screening_Rank",
    "Title",
    "SMILES",
    "Predicted_Activity",
    "Active_Probability"
]

display_columns += [
    feature for feature in MANDATORY_SHAP_FEATURES
    if feature in results_df.columns
]

display_columns = [
    col for col in display_columns
    if col in results_df.columns
]

print("\nTop 10 screened molecules:")

print(
    results_df[display_columns]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 16. SAVE-FILE SUMMARY
# ============================================================

print("\nAll screening results:")
print(ALL_OUTPUT_FILE)

print("\nTop 500 results:")
print(TOP_OUTPUT_FILE)

print("\n" + "=" * 60)
print("END MODULE 13")
print("=" * 60)

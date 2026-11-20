
# ============================================================
# MODULE 8
# EXTERNAL VALIDATION
# RANDOM FOREST QSAR MODEL
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    roc_curve
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

EXTERNAL_DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "external"
    / "External_Dataset_cleaned.csv"
)

MODEL_DIR = PROJECT_DIR / "outputs" / "model"

MODEL_FILE = MODEL_DIR / "rf_qsar_model.pkl"
SCALER_FILE = MODEL_DIR / "rf_scaler.pkl"
FEATURE_NAMES_FILE = MODEL_DIR / "rf_feature_names.pkl"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "validation"

PREDICTIONS_FILE = (
    OUTPUT_DIR / "rf_external_validation_predictions.csv"
)

CONFUSION_MATRIX_FILE = (
    OUTPUT_DIR / "rf_external_confusion_matrix.png"
)

ROC_CURVE_FILE = (
    OUTPUT_DIR / "rf_external_roc_curve.png"
)

TOP_HITS_FILE = (
    OUTPUT_DIR / "rf_top_virtual_screening_hits.csv"
)


# ============================================================
# 2. LOAD EXTERNAL DATASET
# ============================================================

def load_external_dataset():

    if not EXTERNAL_DATA_FILE.exists():
        raise FileNotFoundError(
            f"External dataset not found: {EXTERNAL_DATA_FILE}"
        )

    external_df = pd.read_csv(EXTERNAL_DATA_FILE)

    # Remove accidental pandas index columns.
    external_df = external_df.loc[
        :,
        ~external_df.columns.str.contains("^Unnamed")
    ]

    required_columns = [
        "Molecule_ID",
        "Standard_Value",
        "Standard_Units",
        "SMILES",
        "Label_reg",
        "Label_clf"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in external_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required external dataset columns: "
            f"{missing_columns}"
        )

    if external_df.empty:
        raise ValueError("External dataset is empty.")

    print("\n===================================")
    print("EXTERNAL DATASET")
    print("===================================")

    print("Dataset shape:", external_df.shape)
    print("Columns:", external_df.columns.tolist())

    return external_df


# ============================================================
# 3. PREPARE TRUE ACTIVITY LABELS
# ============================================================

def prepare_activity_labels(external_df):

    external_df["pMIC"] = pd.to_numeric(
        external_df["Label_reg"],
        errors="coerce"
    )

    external_df = external_df[
        external_df["pMIC"].notna()
        & np.isfinite(external_df["pMIC"])
    ].copy()

    if external_df.empty:
        raise ValueError(
            "No valid pMIC values remain in the external dataset."
        )

    external_df["Activity_Class"] = np.where(
        external_df["pMIC"] >= 6,
        1,
        0
    )

    external_df["Activity"] = np.where(
        external_df["Activity_Class"] == 1,
        "ACTIVE",
        "INACTIVE"
    )

    print("\nExternal activity distribution:")
    print(external_df["Activity"].value_counts())

    return external_df


# ============================================================
# 4. VALIDATE SMILES
# ============================================================

def validate_smiles(external_df):

    valid_rows = []
    invalid_smiles = []

    for index, smiles in enumerate(external_df["SMILES"]):

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is not None:
            valid_rows.append(index)
        else:
            invalid_smiles.append(smiles)

    external_df = external_df.iloc[
        valid_rows
    ].reset_index(drop=True)

    print("\n===================================")
    print("SMILES VALIDATION")
    print("===================================")

    print("Valid molecules:", len(external_df))
    print("Invalid SMILES:", len(invalid_smiles))

    if external_df.empty:
        raise ValueError(
            "No valid molecules remain after SMILES validation."
        )

    return external_df


# ============================================================
# 5. GENERATE RDKit DESCRIPTORS
# ============================================================

def generate_descriptors(external_df):

    print("\nGenerating RDKit descriptors...")

    descriptor_names = [
        descriptor[0]
        for descriptor in Descriptors._descList
    ]

    calculator = (
        MoleculeDescriptors.MolecularDescriptorCalculator(
            descriptor_names
        )
    )

    descriptor_data = []

    for smiles in external_df["SMILES"]:

        mol = Chem.MolFromSmiles(smiles)

        descriptors = calculator.CalcDescriptors(mol)

        descriptor_data.append(descriptors)

    descriptor_df = pd.DataFrame(
        descriptor_data,
        columns=descriptor_names
    )

    print("Raw descriptor shape:", descriptor_df.shape)

    return descriptor_df


# ============================================================
# 6. LOAD TRAINED MODEL, SCALER AND FEATURE NAMES
# ============================================================

def load_trained_model():

    required_files = [
        MODEL_FILE,
        SCALER_FILE,
        FEATURE_NAMES_FILE
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required trained model file not found: "
                f"{file_path}"
            )

    rf_model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)

    print("\n===================================")
    print("MODEL LOADED")
    print("===================================")

    print("RF model loaded successfully.")
    print("Training features:", len(feature_names))

    return rf_model, scaler, feature_names


# ============================================================
# 7. MATCH EXTERNAL FEATURES TO TRAINING FEATURES
# ============================================================

def match_training_features(
    descriptor_df,
    feature_names
):

    missing_features = [
        feature
        for feature in feature_names
        if feature not in descriptor_df.columns
    ]

    if missing_features:
        raise ValueError(
            "External descriptor generation is missing "
            f"{len(missing_features)} training features. "
            "Check RDKit versions and descriptor definitions. "
            f"Examples: {missing_features[:10]}"
        )

    # Preserve exactly the feature order used during training.
    descriptor_df = descriptor_df.reindex(
        columns=feature_names
    )

    descriptor_df = descriptor_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    missing_value_count = int(
        descriptor_df.isna().sum().sum()
    )

    if missing_value_count > 0:
        print(
            "\nWarning:",
            missing_value_count,
            "external descriptor values are missing."
        )

        # Retain the original workflow's missing-value handling.
        descriptor_df = descriptor_df.fillna(0)

    print("\nMatched external feature shape:")
    print(descriptor_df.shape)

    print(
        "Remaining NaN values:",
        descriptor_df.isna().sum().sum()
    )

    print(
        "Remaining Inf values:",
        np.isinf(descriptor_df.to_numpy()).sum()
    )

    return descriptor_df


# ============================================================
# 8. SCALE EXTERNAL FEATURES
# ============================================================

def scale_external_features(
    descriptor_df,
    scaler
):

    if descriptor_df.shape[1] != scaler.n_features_in_:
        raise ValueError(
            "External feature count does not match "
            "the number of features expected by the scaler."
        )

    X_external_scaled = scaler.transform(descriptor_df)

    print("\nExternal features scaled successfully.")

    return X_external_scaled


# ============================================================
# 9. GENERATE RF PREDICTIONS
# ============================================================

def generate_predictions(
    external_df,
    X_external_scaled,
    rf_model
):

    predictions = rf_model.predict(X_external_scaled)

    # Locate the probability column corresponding to class 1.
    active_class_indices = np.where(
        rf_model.classes_ == 1
    )[0]

    if len(active_class_indices) != 1:
        raise ValueError(
            "The trained RF model does not contain "
            "the expected ACTIVE class (1)."
        )

    active_class_index = active_class_indices[0]

    probabilities = rf_model.predict_proba(
        X_external_scaled
    )[:, active_class_index]

    external_df["Predicted_Class"] = predictions
    external_df["Probability_ACTIVE"] = probabilities

    external_df["Predicted_Label"] = np.where(
        predictions == 1,
        "ACTIVE",
        "INACTIVE"
    )

    external_df = external_df.sort_values(
        by="Probability_ACTIVE",
        ascending=False
    ).reset_index(drop=True)

    return external_df


# ============================================================
# 10. CALCULATE EXTERNAL VALIDATION METRICS
# ============================================================

def calculate_metrics(external_df):

    y_true = external_df["Activity_Class"]
    y_pred = external_df["Predicted_Class"]
    y_prob = external_df["Probability_ACTIVE"]

    if y_true.nunique() < 2:
        raise ValueError(
            "External validation requires both ACTIVE and "
            "INACTIVE compounds to calculate ROC-AUC."
        )

    auc = roc_auc_score(y_true, y_prob)

    bac = balanced_accuracy_score(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n===================================")
    print("RF EXTERNAL VALIDATION PERFORMANCE")
    print("===================================")

    print(f"\nROC-AUC           : {auc:.4f}")
    print(f"Balanced Accuracy : {bac:.4f}")
    print(f"MCC               : {mcc:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return auc, bac, mcc


# ============================================================
# 11. GENERATE CONFUSION MATRIX
# ============================================================

def generate_confusion_matrix(external_df):

    cm = confusion_matrix(
        external_df["Activity_Class"],
        external_df["Predicted_Class"]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("RF External Validation Confusion Matrix")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=300)
    plt.close()

    print("\nConfusion matrix saved:")
    print(CONFUSION_MATRIX_FILE)


# ============================================================
# 12. GENERATE ROC CURVE
# ============================================================

def generate_roc_curve(external_df, auc):

    y_true = external_df["Activity_Class"]
    y_prob = external_df["Probability_ACTIVE"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RF External Validation ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(ROC_CURVE_FILE, dpi=300)
    plt.close()

    print("\nROC curve saved:")
    print(ROC_CURVE_FILE)


# ============================================================
# 13. SAVE VALIDATION RESULTS AND TOP HITS
# ============================================================

def save_results(external_df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    external_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    top_hits = external_df.head(20)

    top_hits.to_csv(
        TOP_HITS_FILE,
        index=False
    )

    print("\n===================================")
    print("RESULTS SAVED")
    print("===================================")

    print(PREDICTIONS_FILE)
    print(TOP_HITS_FILE)


# ============================================================
# 14. DISPLAY TOP 20 COMPOUNDS
# ============================================================

def display_top_hits(external_df):

    display_columns = [
        "Molecule_ID",
        "SMILES",
        "pMIC",
        "Activity",
        "Predicted_Label",
        "Probability_ACTIVE"
    ]

    display_columns = [
        col for col in display_columns
        if col in external_df.columns
    ]

    print("\n===================================")
    print("TOP 20 COMPOUNDS BY ACTIVE PROBABILITY")
    print("===================================")

    print(
        external_df[display_columns]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 15. FINAL EXTERNAL VALIDATION SUMMARY
# ============================================================

def print_final_summary(external_df, feature_names):

    print("\n===================================")
    print("FINAL EXTERNAL VALIDATION SUMMARY")
    print("===================================")

    print("\nTotal external compounds:", len(external_df))

    print("\nActual activity distribution:")
    print(external_df["Activity"].value_counts())

    print("\nPredicted activity distribution:")
    print(external_df["Predicted_Label"].value_counts())

    print("\nFeature count used:", len(feature_names))


# ============================================================
# 16. MAIN WORKFLOW
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    external_df = load_external_dataset()

    external_df = prepare_activity_labels(external_df)

    external_df = validate_smiles(external_df)

    descriptor_df = generate_descriptors(external_df)

    rf_model, scaler, feature_names = load_trained_model()

    descriptor_df = match_training_features(
        descriptor_df,
        feature_names
    )

    X_external_scaled = scale_external_features(
        descriptor_df,
        scaler
    )

    external_df = generate_predictions(
        external_df,
        X_external_scaled,
        rf_model
    )

    auc, bac, mcc = calculate_metrics(external_df)

    generate_confusion_matrix(external_df)

    generate_roc_curve(external_df, auc)

    save_results(external_df)

    display_top_hits(external_df)

    print_final_summary(
        external_df,
        feature_names
    )

    print("\nModule 8 completed successfully.")


if __name__ == "__main__":
    main()


# ============================================================
# END OF MODULE 8
# ============================================================

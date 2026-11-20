
"""
Module 4: Universal QSAR Data Preprocessing
RF-QSAR-ML Pipeline

Prepares the combined anti-M. tuberculosis activity dataset
for molecular descriptor generation and Random Forest modelling.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Keep the original training dataset locally in the data folder.
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "Training_Dataset_QSAR.csv"

# Processed datasets are written to the outputs folder.
OUTPUT_DIR = PROJECT_DIR / "outputs" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "rf_cleaned_qsar_dataset.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset(input_file):
    """Load the combined QSAR training dataset."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {input_file}\n"
            "Place Training_Dataset_QSAR.csv inside the data folder."
        )

    df = pd.read_csv(input_file)

    print("\nDataset loaded successfully.")
    print("Original dataset shape:", df.shape)

    return df


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """Standardize column names for consistent processing."""

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

def validate_dataset(df):
    """Check whether the required molecular information is available."""

    if "SMILES" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'SMILES' column."
        )

    print("\nSMILES column detected successfully.")


# ============================================================
# 5. STANDARDIZE TARGET ORGANISM
# ============================================================

def add_target_organism(df):
    """
    Retain the biological target as dataset metadata.

    Target_Organism is not intended to be used as a
    molecular machine-learning descriptor.
    """

    df = df.copy()

    df["Target_Organism"] = "Mycobacterium tuberculosis"

    return df


# ============================================================
# 6. PREPARE pMIC VALUES
# ============================================================

def prepare_pmic(df):
    """
    Obtain pMIC using the following priority:

    1. Label_reg
    2. pMIC
    3. MIC
    4. Standard_Value

    For MIC expressed in µM:

        pMIC = 6 - log10(MIC)

    IMPORTANT:
    Standard_Value is assumed to be expressed in µM,
    matching the assumption in the original workflow.
    Verify its units before using this fallback.
    """

    df = df.copy()

    if "Label_reg" in df.columns:

        print("\nLabel_reg detected. Using it as the pMIC source.")

        df["pMIC"] = pd.to_numeric(
            df["Label_reg"],
            errors="coerce"
        )

    elif "pMIC" in df.columns:

        print("\npMIC column detected.")

        df["pMIC"] = pd.to_numeric(
            df["pMIC"],
            errors="coerce"
        )

    elif "MIC" in df.columns:

        print("\nMIC column detected.")
        print("Converting MIC (µM) to pMIC.")

        mic_values = pd.to_numeric(
            df["MIC"],
            errors="coerce"
        )

        valid_mic = mic_values > 0

        df = df.loc[valid_mic].copy()
        mic_values = mic_values.loc[valid_mic]

        df["pMIC"] = 6 - np.log10(mic_values)

    elif "Standard_Value" in df.columns:

        print("\nStandard_Value detected.")
        print(
            "Assuming Standard_Value is expressed in µM "
            "for pMIC conversion."
        )

        standard_values = pd.to_numeric(
            df["Standard_Value"],
            errors="coerce"
        )

        valid_values = standard_values > 0

        df = df.loc[valid_values].copy()
        standard_values = standard_values.loc[valid_values]

        df["pMIC"] = 6 - np.log10(standard_values)

    else:

        raise ValueError(
            "No activity target detected. Dataset must contain "
            "Label_reg, pMIC, MIC, or Standard_Value."
        )

    # Remove missing and infinite pMIC values.
    before = len(df)

    df["pMIC"] = df["pMIC"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(subset=["pMIC"]).copy()

    print("\nInvalid pMIC rows removed:", before - len(df))
    print("Valid activity records:", len(df))

    return df


# ============================================================
# 7. REMOVE INVALID SMILES
# ============================================================

def clean_smiles(df):
    """Remove records with missing or empty SMILES strings."""

    df = df.copy()

    before = len(df)

    df = df.dropna(subset=["SMILES"]).copy()

    df["SMILES"] = (
        df["SMILES"]
        .astype(str)
        .str.strip()
    )

    df = df[df["SMILES"] != ""].copy()

    print("\nInvalid/empty SMILES removed:", before - len(df))
    print("Valid SMILES records:", len(df))

    return df


# ============================================================
# 8. CREATE ACTIVITY LABELS
# ============================================================

def create_activity_labels(df):
    """
    Define activity using the established pMIC threshold.

    pMIC >= 6 : ACTIVE
    pMIC <  6 : INACTIVE

    Activity_Class and Label_clf are numerical
    representations of the same binary classification target.
    """

    df = df.copy()

    df["Activity"] = np.where(
        df["pMIC"] >= 6,
        "ACTIVE",
        "INACTIVE"
    )

    df["Activity_Class"] = np.where(
        df["pMIC"] >= 6,
        1,
        0
    )

        # Check consistency with the original classifier labels.
    if "Label_clf" in df.columns:

        original_labels = pd.to_numeric(
            df["Label_clf"],
            errors="coerce"
        )

        mismatch = (
            original_labels.notna()
            & (original_labels != df["Activity_Class"])
        )

        if mismatch.any():
            raise ValueError(
                f"{mismatch.sum()} records have Label_clf values "
                "inconsistent with the pMIC >= 6 activity threshold."
            )

    else:
        df["Label_clf"] = df["Activity_Class"]
    return df


# ============================================================
# 9. DATASET QUALITY CHECKS
# ============================================================

def perform_quality_checks(df):
    """Summarize activity, pMIC, duplicates, and missing values."""

    print("\n" + "=" * 45)
    print("ACTIVITY DISTRIBUTION")
    print("=" * 45)

    print(df["Activity"].value_counts())

    print("\nActivity proportions:")
    print(df["Activity_Class"].value_counts(normalize=True))

    print("\n" + "=" * 45)
    print("pMIC STATISTICS")
    print("=" * 45)

    print(df["pMIC"].describe())

    print("\n" + "=" * 45)
    print("DUPLICATE CHECK")
    print("=" * 45)

    duplicate_count = df.duplicated().sum()

    print("Exact duplicate rows found:", duplicate_count)
    print("No duplicate removal applied.")
    print(
        "Repeated molecules with different experimental "
        "measurements are retained."
    )

    print("\n" + "=" * 45)
    print("MISSING VALUE SUMMARY")
    print("=" * 45)

    missing_summary = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing_summary[missing_summary > 0])


# ============================================================
# 10. SAVE PROCESSED DATASET
# ============================================================

def save_dataset(df, output_file):
    """Save the cleaned QSAR dataset."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\nProcessed dataset saved successfully:")
    print(output_file)


# ============================================================
# 11. MAIN PREPROCESSING WORKFLOW
# ============================================================

def main():

    print("=" * 55)
    print("RF-QSAR DATA PREPROCESSING")
    print("=" * 55)

    df = load_dataset(INPUT_FILE)

    df = clean_column_names(df)

    validate_dataset(df)

    df = add_target_organism(df)

    df = prepare_pmic(df)

    df = clean_smiles(df)

    df = create_activity_labels(df)

    perform_quality_checks(df)

    save_dataset(df, OUTPUT_FILE)

    preview_columns = [
        "SMILES",
        "pMIC",
        "Activity",
        "Activity_Class",
        "Label_clf",
        "Target_Organism"
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in df.columns
    ]

    print("\nFinal dataset preview:")
    print(df[preview_columns].head(10))

    print("\nFinal dataset shape:", df.shape)

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()

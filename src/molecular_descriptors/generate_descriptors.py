
"""
Module 5: Molecular Descriptor Generation
RF-QSAR-ML Pipeline

Generates RDKit molecular descriptors for the cleaned
anti-Mycobacterium tuberculosis QSAR dataset.

The molecular descriptors are kept separate from activity
labels and other dataset metadata.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "processed"
    / "rf_cleaned_qsar_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "descriptors"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "rf_qsar_descriptors_dataset.csv"
)

DESCRIPTOR_LIST_FILE = (
    OUTPUT_DIR
    / "rdkit_descriptor_names.txt"
)


# ============================================================
# 2. LOAD CLEANED DATASET
# ============================================================

def load_dataset(input_file):
    """Load the preprocessed QSAR dataset."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_file}\n"
            "Run Module 4 first and check its output location."
        )

    df = pd.read_csv(input_file)

    print("\nCleaned dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    if "SMILES" not in df.columns:
        raise ValueError(
            "Input dataset must contain a 'SMILES' column."
        )

    if "pMIC" not in df.columns:
        raise ValueError(
            "Input dataset must contain a 'pMIC' column."
        )

    return df


# ============================================================
# 3. REMOVE INVALID ACTIVITY VALUES
# ============================================================

def clean_activity_values(df):
    """Remove records with missing or infinite pMIC values."""

    df = df.copy()

    before = len(df)

    df["pMIC"] = pd.to_numeric(
        df["pMIC"],
        errors="coerce"
    )

    df["pMIC"] = df["pMIC"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=["pMIC"]
    ).copy()

    print("\nInvalid pMIC rows removed:", before - len(df))

    return df


# ============================================================
# 4. VALIDATE SMILES
# ============================================================

def validate_smiles(df):
    """
    Retain records with valid RDKit-readable SMILES.

    Original SMILES strings are preserved.
    No canonicalization or duplicate removal is performed.
    """

    df = df.copy()

    valid_mask = []

    for smiles in df["SMILES"]:

        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        valid_mask.append(
            mol is not None
        )

    before = len(df)

    df = df.loc[valid_mask].copy()

    print("\nInvalid SMILES removed:", before - len(df))
    print("Valid molecules:", len(df))

    return df


# ============================================================
# 5. GENERATE RDKIT DESCRIPTORS
# ============================================================

def generate_descriptors(df):
    """Calculate the complete RDKit descriptor set."""

    print("\nGenerating molecular descriptors...")

    descriptor_names = [
        descriptor[0]
        for descriptor in Descriptors._descList
    ]

    calculator = (
        MoleculeDescriptors.MolecularDescriptorCalculator(
            descriptor_names
        )
    )

    descriptor_list = []

    for smiles in df["SMILES"]:

        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        descriptor_values = calculator.CalcDescriptors(
            mol
        )

        descriptor_list.append(
            descriptor_values
        )

    descriptor_df = pd.DataFrame(
        descriptor_list,
        columns=descriptor_names,
        index=df.index
    )

    print("RDKit version:", rdBase.rdkitVersion)
    print("Number of descriptors generated:", len(descriptor_names))
    print("Descriptor matrix shape:", descriptor_df.shape)

    return descriptor_df, descriptor_names


# ============================================================
# 6. REMOVE INVALID DESCRIPTOR FEATURES
# ============================================================

def clean_descriptor_features(descriptor_df):
    """
    Retain numeric descriptors without missing or infinite values.

    Only molecular descriptor columns are evaluated here.
    Activity labels and metadata are not treated as features.
    """

    descriptor_df = descriptor_df.copy()

    descriptor_df = descriptor_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    descriptor_df = descriptor_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    invalid_columns = descriptor_df.columns[
        descriptor_df.isnull().any()
    ].tolist()

    descriptor_df = descriptor_df.dropna(
        axis=1
    )

    print("\nInvalid descriptor columns removed:", len(invalid_columns))

    if invalid_columns:
        print("Removed descriptor names:")
        print(invalid_columns)

    print("Retained descriptor shape:", descriptor_df.shape)

    return descriptor_df


# ============================================================
# 7. COMBINE DESCRIPTORS WITH DATASET METADATA
# ============================================================

def assemble_qsar_dataset(df, descriptor_df):
    """
    Combine molecular descriptors with essential metadata
    and activity labels.

    Metadata and target labels are retained separately from
    the molecular descriptor matrix.
    """

    metadata_columns = [
        "SMILES",
        "pMIC",
        "Activity",
        "Activity_Class",
        "Label_clf",
        "Target_Organism"
    ]

    metadata_columns = [
        column
        for column in metadata_columns
        if column in df.columns
    ]

    metadata_df = df[
        metadata_columns
    ].copy()

    qsar_df = pd.concat(
        [
            metadata_df,
            descriptor_df
        ],
        axis=1
    )

    print("\nFinal QSAR dataset shape:", qsar_df.shape)

    return qsar_df


# ============================================================
# 8. SAVE DESCRIPTOR INFORMATION
# ============================================================

def save_outputs(qsar_df, descriptor_names):
    """Save the descriptor dataset and descriptor-name record."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    qsar_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    with open(
        DESCRIPTOR_LIST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(descriptor_names)
        )

    print("\nDescriptor dataset saved:")
    print(OUTPUT_FILE)

    print("\nRDKit descriptor-name record saved:")
    print(DESCRIPTOR_LIST_FILE)


# ============================================================
# 9. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 55)
    print("RF-QSAR MOLECULAR DESCRIPTOR GENERATION")
    print("=" * 55)

    df = load_dataset(INPUT_FILE)

    df = clean_activity_values(df)

    df = validate_smiles(df)

    descriptor_df, descriptor_names = generate_descriptors(df)

    descriptor_df = clean_descriptor_features(
        descriptor_df
    )

    qsar_df = assemble_qsar_dataset(
        df,
        descriptor_df
    )

    save_outputs(
        qsar_df,
        descriptor_df.columns.tolist()
    )

    print("\nFinal dataset preview:")
    print(qsar_df.head())

    print("\nFinal dataset shape:")
    print(qsar_df.shape)

    print("\nDescriptor generation completed successfully.")


if __name__ == "__main__":
    main()

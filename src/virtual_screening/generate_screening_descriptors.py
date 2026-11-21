
# ============================================================
# MODULE 12
# GENERIC DESCRIPTOR GENERATION FOR VIRTUAL SCREENING
# SCREENING LIBRARY -> TRAINING-MATCHED RDKit DESCRIPTORS
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS AND LIBRARY CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Change this name for each new screening library.
LIBRARY_NAME = "Screening Library"

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "screening_libraries"
    / f"{LIBRARY_NAME}.csv"
)

FEATURE_FILE = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection"
    / "selected_descriptor_names.txt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "virtual_screening"
    / LIBRARY_NAME
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"{LIBRARY_NAME}_descriptor_dataset.csv"
)

INVALID_FILE = (
    OUTPUT_DIR
    / f"{LIBRARY_NAME}_invalid_smiles.csv"
)


# ============================================================
# 2. LOAD TRAINING DESCRIPTOR NAMES
# ============================================================

print("=" * 60)
print("MODULE 12: GENERIC SCREENING DESCRIPTOR GENERATION")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Screening library not found:\n{INPUT_FILE}"
    )

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Selected descriptor list not found:\n{FEATURE_FILE}"
    )

descriptor_names = [
    line.strip()
    for line in FEATURE_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if not descriptor_names:
    raise ValueError("The selected descriptor list is empty.")

if len(descriptor_names) != len(set(descriptor_names)):
    raise ValueError("Duplicate names found in the descriptor list.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLibrary:", LIBRARY_NAME)
print("Input:", INPUT_FILE)
print("Selected descriptors:", len(descriptor_names))


# ============================================================
# 3. LOAD SCREENING LIBRARY
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nLibrary loaded:", df.shape)

if "SMILES" not in df.columns:
    raise ValueError("Required SMILES column is missing.")

# Preserve the original row number for traceability.
df.insert(0, "Original_Row_Index", np.arange(len(df)))

# Add source filename if it is not already present.
if "Source_File" not in df.columns:
    df["Source_File"] = INPUT_FILE.name


# ============================================================
# 4. INITIALIZE RDKit DESCRIPTOR CALCULATOR
# ============================================================

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
    descriptor_names
)

# Verify that every requested descriptor is supported by RDKit.
available_descriptors = set(calculator.GetDescriptorNames())

missing_descriptors = [
    name for name in descriptor_names
    if name not in available_descriptors
]

if missing_descriptors:
    raise ValueError(
        "The following selected descriptors are unavailable in this "
        f"RDKit installation:\n{missing_descriptors}"
    )


# ============================================================
# 5. VALIDATE SMILES AND GENERATE DESCRIPTORS
# ============================================================

valid_records = []
invalid_records = []

print("\nValidating molecules and calculating descriptors...")

for _, row in df.iterrows():

    smiles = row["SMILES"]

    if pd.isna(smiles) or not str(smiles).strip():
        invalid_records.append(row.to_dict())
        continue

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        invalid_records.append(row.to_dict())
        continue

    try:
        values = calculator.CalcDescriptors(mol)

        descriptor_values = np.asarray(values, dtype=float)

        if not np.isfinite(descriptor_values).all():
            raise ValueError("Non-finite descriptor value encountered.")

        record = row.to_dict()

        for name, value in zip(descriptor_names, descriptor_values):
            record[name] = value

        valid_records.append(record)

    except Exception as exc:
        raise RuntimeError(
            "Descriptor calculation failed for "
            f"original row {row['Original_Row_Index']} "
            f"(SMILES: {smiles}). Error: {exc}"
        ) from exc


# ============================================================
# 6. CREATE FINAL DATASETS
# ============================================================

if not valid_records:
    raise ValueError(
        "No valid molecules remain after SMILES validation."
    )

screening_df = pd.DataFrame(valid_records)

invalid_df = pd.DataFrame(invalid_records)

# Ensure the descriptor columns retain training order.
metadata_columns = [
    col for col in screening_df.columns
    if col not in descriptor_names
]

screening_df = screening_df[
    metadata_columns + descriptor_names
]


# ============================================================
# 7. FINAL VALIDATION
# ============================================================

if screening_df[descriptor_names].isna().any().any():
    raise ValueError("Missing values detected in generated descriptors.")

if not np.isfinite(
    screening_df[descriptor_names].to_numpy(dtype=float)
).all():
    raise ValueError("Non-finite values detected in generated descriptors.")


# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

screening_df.to_csv(OUTPUT_FILE, index=False)

if not invalid_df.empty:
    invalid_df.to_csv(INVALID_FILE, index=False)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODULE 12 COMPLETE")
print("=" * 60)

print("Total input molecules:", len(df))
print("Valid molecules:", len(screening_df))
print("Invalid molecules:", len(invalid_df))
print("Training-matched descriptors:", len(descriptor_names))

print("\nDescriptor dataset:")
print(OUTPUT_FILE)

if not invalid_df.empty:
    print("\nInvalid SMILES report:")
    print(INVALID_FILE)

print("=" * 60)

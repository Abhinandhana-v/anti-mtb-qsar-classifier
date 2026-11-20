
# ============================================================
# MODULE 6
# FEATURE CLEANING + FEATURE SELECTION
# RANDOM FOREST QSAR PIPELINE
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "descriptors"
    / "rf_qsar_descriptors_dataset.csv"
)

DESCRIPTOR_LIST_FILE = (
    PROJECT_DIR
    / "outputs"
    / "descriptors"
    / "rdkit_descriptor_names.txt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection"
)

OUTPUT_FILE = OUTPUT_DIR / "rf_final_qsar_dataset.csv"

SELECTED_FEATURES_FILE = (
    OUTPUT_DIR / "selected_descriptor_names.txt"
)


# ============================================================
# 2. LOAD DATASET AND DESCRIPTOR NAMES
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Descriptor dataset not found: {INPUT_FILE}"
        )

    if not DESCRIPTOR_LIST_FILE.exists():
        raise FileNotFoundError(
            f"RDKit descriptor list not found: "
            f"{DESCRIPTOR_LIST_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    descriptor_names = [
        line.strip()
        for line in DESCRIPTOR_LIST_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if df.empty:
        raise ValueError("The input dataset is empty.")

    print("\nDataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df, descriptor_names


# ============================================================
# 3. IDENTIFY DESCRIPTOR AND METADATA COLUMNS
# ============================================================

def separate_descriptors(df, descriptor_names):

    descriptor_cols = [
        col for col in descriptor_names
        if col in df.columns
    ]

    if not descriptor_cols:
        raise ValueError(
            "No RDKit descriptor columns were found "
            "in the input dataset."
        )

    missing_descriptors = [
        col for col in descriptor_names
        if col not in df.columns
    ]

    if missing_descriptors:
        print(
            "\nWarning: Some descriptors listed in the "
            "descriptor file are absent from the dataset:",
            len(missing_descriptors)
        )

    # Preserve all non-descriptor columns as metadata.
    metadata_cols = [
        col for col in df.columns
        if col not in descriptor_cols
    ]

    X = df[descriptor_cols].copy()

    print("\nInitial RDKit descriptor count:", X.shape[1])
    print("Metadata columns retained:", len(metadata_cols))

    return X, metadata_cols


# ============================================================
# 4. KEEP NUMERIC DESCRIPTORS ONLY
# ============================================================

def keep_numeric_descriptors(X):

    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError(
            "No numeric descriptor columns remain."
        )

    print("Numeric descriptor count:", X.shape[1])

    return X


# ============================================================
# 5. REMOVE INF / NaN VALUES
# ============================================================

def clean_missing_values(X):

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    before = X.shape[1]

    X = X.dropna(axis=1)

    removed = before - X.shape[1]

    print(
        "\nDescriptors removed due to NaN/INF:",
        removed
    )

    print(
        "Descriptors after missing-value cleaning:",
        X.shape[1]
    )

    if X.shape[1] == 0:
        raise ValueError(
            "No descriptors remain after missing-value cleaning."
        )

    return X


# ============================================================
# 6. REMOVE ZERO-VARIANCE FEATURES
# ============================================================

def remove_constant_features(X):

    print("\nRemoving constant descriptors...")

    before = X.shape[1]

    variance_filter = VarianceThreshold(threshold=0)

    X_filtered = variance_filter.fit_transform(X)

    selected_cols = X.columns[
        variance_filter.get_support()
    ]

    X = pd.DataFrame(
        X_filtered,
        columns=selected_cols,
        index=X.index
    )

    removed = before - X.shape[1]

    print("Constant descriptors removed:", removed)
    print("Descriptors after variance filtering:", X.shape[1])

    if X.shape[1] == 0:
        raise ValueError(
            "No descriptors remain after variance filtering."
        )

    return X, removed


# ============================================================
# 7. REMOVE HIGHLY CORRELATED FEATURES
# ============================================================

def remove_correlated_features(X):

    print("\nRemoving highly correlated descriptors...")

    correlation_matrix = X.corr().abs()

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    to_drop = [
        column
        for column in upper_triangle.columns
        if (upper_triangle[column] > 0.90).any()
    ]

    X = X.drop(columns=to_drop)

    print(
        "Highly correlated descriptors removed:",
        len(to_drop)
    )

    print(
        "Remaining descriptors:",
        X.shape[1]
    )

    if X.shape[1] == 0:
        raise ValueError(
            "No descriptors remain after correlation filtering."
        )

    return X, to_drop


# ============================================================
# 8. REBUILD FINAL QSAR DATASET
# ============================================================

def rebuild_dataset(df, metadata_cols, X):

    final_df = pd.concat(
        [
            df[metadata_cols].reset_index(drop=True),
            X.reset_index(drop=True)
        ],
        axis=1
    )

    print("\nFinal dataset shape:", final_df.shape)

    return final_df


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

def save_outputs(final_df, X):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    SELECTED_FEATURES_FILE.write_text(
        "\n".join(X.columns),
        encoding="utf-8"
    )

    print("\nSaved final QSAR dataset:")
    print(OUTPUT_FILE)

    print("\nSaved selected descriptor names:")
    print(SELECTED_FEATURES_FILE)


# ============================================================
# 10. FEATURE SELECTION SUMMARY
# ============================================================

def print_summary(
    original_count,
    numeric_count,
    cleaned_count,
    variance_removed,
    correlation_removed,
    final_count,
    final_df
):

    print("\n===================================")
    print("FEATURE SELECTION SUMMARY")
    print("===================================")

    print("Initial RDKit descriptors:", original_count)
    print("Numeric descriptors:", numeric_count)
    print("After NaN/INF cleaning:", cleaned_count)
    print("Constant descriptors removed:", variance_removed)
    print("Highly correlated descriptors removed:", correlation_removed)
    print("Final descriptors:", final_count)

    print("\nFinal dataset shape:", final_df.shape)

    print("\nFinal dataset columns:")
    print(final_df.columns.tolist())

    print("\nFinal dataset preview:")
    print(final_df.head())


# ============================================================
# 11. MAIN WORKFLOW
# ============================================================

def main():

    df, descriptor_names = load_dataset()

    X, metadata_cols = separate_descriptors(
        df,
        descriptor_names
    )

    original_count = X.shape[1]

    X = keep_numeric_descriptors(X)
    numeric_count = X.shape[1]

    X = clean_missing_values(X)
    cleaned_count = X.shape[1]

    X, variance_removed = remove_constant_features(X)

    X, correlated_features = remove_correlated_features(X)

    final_df = rebuild_dataset(
        df,
        metadata_cols,
        X
    )

    save_outputs(final_df, X)

    print_summary(
        original_count,
        numeric_count,
        cleaned_count,
        variance_removed,
        len(correlated_features),
        X.shape[1],
        final_df
    )


if __name__ == "__main__":
    main()

# ============================================================
# END OF MODULE 6
# ============================================================

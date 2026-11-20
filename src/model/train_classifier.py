
# ============================================================
# MODULE 7
# RANDOM FOREST QSAR-ML MODEL BUILDING
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection"
    / "rf_final_qsar_dataset.csv"
)

FEATURE_LIST_FILE = (
    PROJECT_DIR
    / "outputs"
    / "feature_selection"
    / "selected_descriptor_names.txt"
)

MODEL_DIR = PROJECT_DIR / "outputs" / "model"

TRAINING_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "model_training"
)

FIGURE_DIR = (
    PROJECT_DIR / "outputs" / "figures"
)


# ============================================================
# 2. LOAD FEATURE-SELECTED DATASET
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature-selected dataset not found: {INPUT_FILE}"
        )

    if not FEATURE_LIST_FILE.exists():
        raise FileNotFoundError(
            f"Selected descriptor list not found: {FEATURE_LIST_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    feature_names = [
        line.strip()
        for line in FEATURE_LIST_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if df.empty:
        raise ValueError("The input dataset is empty.")

    if not feature_names:
        raise ValueError("No selected descriptor names were found.")

    print("\nDataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("Selected descriptors:", len(feature_names))

    return df, feature_names


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features_and_target(df, feature_names):

    target_column = "Activity_Class"

    if target_column not in df.columns:
        raise ValueError(
            "Activity_Class column is required for classification."
        )

    missing_features = [
        col for col in feature_names
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Selected descriptors missing from dataset: "
            f"{missing_features[:10]}"
        )

    # Use only the descriptors selected in Module 6.
    X = df[feature_names].copy()

    y = df[target_column].copy()

    # Ensure all model features are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.isna().any().any():
        raise ValueError(
            "Missing or infinite values detected in selected "
            "descriptors. Please check Module 6 outputs."
        )

    if y.isna().any():
        raise ValueError(
            "Missing values detected in Activity_Class."
        )

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Activity_Class must contain only binary labels 0 and 1."
        )

    y = y.astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "Both activity classes (0 and 1) are required."
        )

    print("\nFeature matrix shape:", X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget proportions:")
    print(y.value_counts(normalize=True))

    print("\nNumber of model features:", X.shape[1])

    return X, y


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\n===================================")
    print("TRAIN / TEST SPLIT")
    print("===================================")

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    print("Training feature shape:", X_train.shape)
    print("Testing feature shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ============================================================
# 5. FEATURE SCALING
# ============================================================

def scale_features(X_train, X_test):

    print("\nScaling descriptors...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print("Scaling completed.")

    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# 6. RANDOM FOREST CLASSIFIER
# ============================================================

def train_random_forest(X_train_scaled, y_train):

    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    print("\n===================================")
    print("TRAINING RANDOM FOREST QSAR MODEL")
    print("===================================")

    rf_model.fit(
        X_train_scaled,
        y_train
    )

    print("\nTraining completed successfully.")

    return rf_model


# ============================================================
# 7. TRAINING PROBABILITY SUMMARY
# ============================================================

def training_probability_summary(
    rf_model,
    X_train_scaled
):

    train_prob = rf_model.predict_proba(
        X_train_scaled
    )[:, 1]

    train_pred = rf_model.predict(
        X_train_scaled
    )

    print("\n===================================")
    print("TRAINING SET PROBABILITY SUMMARY")
    print("===================================")

    print("Maximum probability:", train_prob.max())
    print("Minimum probability:", train_prob.min())
    print("Mean probability:", train_prob.mean())

    print("\nPredicted class counts:")
    print(pd.Series(train_pred).value_counts())

    print("\nProbability summary:")
    print(pd.Series(train_prob).describe())

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(6, 4))

    plt.hist(
        train_prob,
        bins=50,
        edgecolor="black"
    )

    plt.xlabel("Probability of Active Class")
    plt.ylabel("Number of Molecules")
    plt.title("Training Probability Distribution")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "rf_training_probability_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 8. TEST SET PREDICTIONS
# ============================================================

def predict_test_set(
    rf_model,
    X_test_scaled
):

    y_pred = rf_model.predict(
        X_test_scaled
    )

    y_prob = rf_model.predict_proba(
        X_test_scaled
    )[:, 1]

    return y_pred, y_prob


# ============================================================
# 9. MODEL PERFORMANCE METRICS
# ============================================================

def evaluate_model(y_test, y_pred, y_prob):

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    bac = balanced_accuracy_score(
        y_test,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    print("\n===================================")
    print("RF-QSAR MODEL PERFORMANCE")
    print("===================================")

    print(f"\nROC-AUC           : {auc:.4f}")
    print(f"Balanced Accuracy : {bac:.4f}")
    print(f"MCC               : {mcc:.4f}")

    print("\nClassification Report:\n")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    return auc, bac, mcc


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_test, y_pred):

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues",
        values_format="d",
        ax=ax
    )

    ax.set_title("RF-QSAR Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "rf_confusion_matrix.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 11. ROC CURVE
# ============================================================

def plot_roc_curve(y_test, y_prob, auc):

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RF-QSAR ROC Curve")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "rf_roc_curve.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    rf_model,
    feature_names
):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    TRAINING_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_df.to_csv(
        TRAINING_OUTPUT_DIR / "rf_feature_importance.csv",
        index=False
    )

    print("\n===================================")
    print("TOP 20 RF FEATURE IMPORTANCES")
    print("===================================")

    print(
        importance_df.head(20).to_string(index=False)
    )

    return importance_df


# ============================================================
# 13. SAVE MODEL AND SCALER
# ============================================================

def save_model(
    rf_model,
    scaler,
    feature_names
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        rf_model,
        MODEL_DIR / "rf_qsar_model.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "rf_scaler.pkl"
    )

    joblib.dump(
        feature_names,
        MODEL_DIR / "rf_feature_names.pkl"
    )

    print("\nModel, scaler and feature names saved.")


# ============================================================
# 14. SAVE TRAINING AND TEST DATA
# ============================================================

def save_training_data(
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    y_prob
):

    TRAINING_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train.to_csv(
        TRAINING_OUTPUT_DIR / "rf_X_train_final.csv",
        index=False
    )

    X_test.to_csv(
        MODEL_DIR / "rf_X_test_final.csv",
        index=False
    )

    y_train.to_csv(
        TRAINING_OUTPUT_DIR / "rf_y_train_final.csv",
        index=False
    )

    y_test.to_csv(
        MODEL_DIR / "rf_y_test_final.csv",
        index=False
    )

    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Probability_ACTIVE": y_prob
    })

    results_df.to_csv(
        TRAINING_OUTPUT_DIR / "rf_test_predictions.csv",
        index=False
    )

    print("\nTraining and test data saved.")


# ============================================================
# 15. SAVE PERFORMANCE METRICS
# ============================================================

def save_metrics(auc, bac, mcc):

    metrics_df = pd.DataFrame({
        "Metric": [
            "ROC-AUC",
            "Balanced Accuracy",
            "MCC"
        ],
        "Value": [
            auc,
            bac,
            mcc
        ]
    })

    metrics_df.to_csv(
        TRAINING_OUTPUT_DIR / "rf_model_metrics.csv",
        index=False
    )


# ============================================================
# 16. MAIN WORKFLOW
# ============================================================

def main():

    df, feature_names = load_dataset()

    X, y = prepare_features_and_target(
        df,
        feature_names
    )

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_test
    )

    rf_model = train_random_forest(
        X_train_scaled,
        y_train
    )

    training_probability_summary(
        rf_model,
        X_train_scaled
    )

    y_pred, y_prob = predict_test_set(
        rf_model,
        X_test_scaled
    )

    auc, bac, mcc = evaluate_model(
        y_test,
        y_pred,
        y_prob
    )

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    plot_roc_curve(
        y_test,
        y_prob,
        auc
    )

    calculate_feature_importance(
        rf_model,
        feature_names
    )

    save_model(
        rf_model,
        scaler,
        feature_names
    )

    save_training_data(
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_prob
    )

    save_metrics(
        auc,
        bac,
        mcc
    )

    print("\n===================================")
    print("MODULE 7 COMPLETE")
    print("===================================")

    print("Model: Random Forest Classifier")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Model features:", len(feature_names))

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Balanced Accuracy: {bac:.4f}")
    print(f"MCC: {mcc:.4f}")

    print("\nModel files saved in:", MODEL_DIR)
    print("Training outputs saved in:", TRAINING_OUTPUT_DIR)
    print("Figures saved in:", FIGURE_DIR)


if __name__ == "__main__":
    main()

# ============================================================
# END OF MODULE 7
# ============================================================

# Anti-Mtb QSAR Classifier

A machine learning-based approach for predicting whole-cell anti-tuberculosis activity using molecular descriptors and Random Forest classification.

## Project Overview

This project focuses on developing a Quantitative Structure-Activity Relationship (QSAR) classification model to predict the anti-tuberculosis activity of small molecules against *Mycobacterium tuberculosis*.

The workflow integrates molecular data preprocessing, RDKit-based molecular descriptor generation, feature selection, Random Forest classification, model validation, SHAP-based explainability, applicability domain-related analysis, and virtual screening.

The classifier is designed to predict whole-cell anti-tuberculosis activity. It does not directly predict molecular binding or inhibition of a specific protein target.

## Workflow

1. **Data preprocessing:** Cleaning and preparation of the training dataset.
2. **Molecular descriptor generation:** Calculation of molecular descriptors using RDKit.
3. **Feature selection:** Removal of zero-variance and highly correlated descriptors.
4. **Model development:** Training a Random Forest classifier.
5. **Model validation:** Internal testing, cross-validation, and external validation.
6. **Model interpretation:** SHAP-based analysis of important molecular descriptors.
7. **Applicability domain-related analysis:** Assessment of prediction reliability using ensemble disagreement.
8. **Virtual screening:** Descriptor generation and activity prediction for external compound libraries.

## Repository Structure

```text
anti-mtb-qsar-classifier/
├── data/
│   ├── raw/
│   ├── external/
│   └── screening_libraries/
├── docs/
├── notebooks/
├── src/
│   ├── data_preprocessing/
│   ├── molecular_descriptors/
│   ├── feature_selection/
│   ├── model/
│   ├── validation/
│   ├── interpretation/
│   ├── applicability_domain/
│   └── virtual_screening/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.11
- RDKit
- NumPy
- pandas
- scikit-learn
- joblib
- Matplotlib
- SHAP

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the Workflow

The Python modules are organized in the `src/` directory and are intended to be executed in the following order:

1. `data_preprocessing/preprocess_data.py`
2. `molecular_descriptors/generate_descriptors.py`
3. `feature_selection/select_features.py`
4. `model/train_classifier.py`
5. `validation/external_validation.py`
6. `validation/cross_validation.py`
7. `interpretation/shap_explainability.py`
8. `applicability_domain/applicability_analysis.py`
9. `virtual_screening/generate_screening_descriptors.py`
10. `virtual_screening/predict_activity.py`

Virtual screening requires a separate screening-library input file. Its availability and redistribution are subject to the applicable data permissions.

## Reproducibility

The repository includes the scripts and dependency specifications required to reproduce the computational workflow. Dataset availability and model outputs will be documented separately.

## Current Status

The repository is under development. Code review, workflow integration, and execution testing are ongoing.

## License

A license will be specified when the repository is prepared for public release.

# Prediction of Surgical Durations Using Machine Learning Methods

This repository contains a reproducible machine learning project for predicting surgical procedure durations from structured healthcare data.

The reusable Python code is organized in `src/`. A small synthetic dataset is provided so that the project can be run without sharing private hospital data.

## Project Motivation

Accurate surgical duration prediction can support operating room planning, staff scheduling, and hospital resource allocation. Surgical durations are difficult to predict because they vary by procedure type, medical unit, clinical context, and staff-related factors.

This project demonstrates a simple and interpretable machine learning workflow for this problem.

## What This Repository Includes

- Data preparation for structured surgical records
- Feature engineering for categorical and numerical variables
- XGBoost regression model
- Stratified analysis of short, medium, and long surgeries
- Model evaluation using MAE, RMSE, and median absolute error
- Example prediction interface for new surgical cases
- A synthetic sample dataset for reproducibility
- Notes connecting the project to statistical bioinformatics research

## Repository Structure

```text
surgical-duration-prediction-ml/
  README.md
  requirements.txt
  .gitignore
  LICENSE

  src/
    data_preprocessing.py
    features.py
    models.py
    evaluation.py
    prediction_interface.py
    train.py

  data/
    README.md
    sample_synthetic_surgeries.csv

  reports/
    thesis_summary.md
    future_work_statistical_bioinformatics.md
    figures/
```

## Data Privacy

The original hospital dataset is not included in this repository because it may contain sensitive or private healthcare information.

Only a small synthetic dataset is provided in `data/sample_synthetic_surgeries.csv`. This file is artificial and is included only to demonstrate how the code works.

Do not upload real patient, hospital, staff, or surgical records to GitHub.

## How to Run

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

Then run the example training workflow from the project root:

```bash
python -m src.train
```

This will:

- load the synthetic sample dataset
- prepare the features
- split surgeries into short, medium, and long duration groups for stratified evaluation
- train an XGBoost model
- evaluate prediction performance
- save example outputs under `reports/`

## Example Methods

The current reproducible example uses:

- one-hot encoding for categorical variables
- median imputation for numerical variables
- XGBoost regression
- train/test evaluation stratified by short, medium, and long surgeries
- subgroup error analysis by surgery duration group

The thesis workflow splits procedures into short, medium, and long surgeries to evaluate whether model performance changes across different duration groups.

## Connection to Statistical Bioinformatics

Although this thesis project focuses on surgical duration prediction rather than omics data, it is connected to statistical bioinformatics through shared methodological ideas:

- modeling heterogeneous biomedical data
- evaluating model performance across meaningful subgroups
- building reproducible analysis pipelines
- using interpretable machine learning methods
- assessing prediction errors and model fit

Future extensions could adapt this workflow to high-dimensional biomedical datasets such as transcriptomics, proteomics, metabolomics, or longitudinal clinical measurements.

## Main Thesis Title

Prediction of Surgical Durations Using Machine Learning Methods

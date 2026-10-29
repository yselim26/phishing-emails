# phishing-emails
COSC 4371 Project - Phishing Email Detection Across Datasets
# Phishing Email Detection Across Datasets

**Course:** COSC 4371 — Security Analytics  
**Project:** Project 5 Checkpoint

## Overview

This project investigates how well a phishing email detection model generalizes across different email datasets. It uses TF-IDF to convert email text into numerical features and logistic regression to classify emails as legitimate or phishing/spam.

Three datasets are used:
- Phishing_Email
- CEAS-08
- IBM email dataset

## Project Files

| File | Description |
|---|---|
| `prepare_data.py` | Prepares and cleans the datasets |
| `check_duplicates.py` | Checks for duplicate emails |
| `check_overlap.py` | Checks for overlap between datasets |
| `make_split.py` | Creates the prepared data and train/validation/test splits |
| `train_model.py` | Trains and evaluates models within each dataset |
| `cross_dataset.py` | Evaluates models trained on one dataset and tested on another |
| `cross_dataset_results.csv` | Stores cross-dataset evaluation results |
| `cross_dataset_auroc.png` | Visualizes cross-dataset AUROC results |
| `split.csv` | Records dataset assignments, labels, and splits |
| `scores.csv` | Stores prediction scores for the checkpoint checker |
| `check_checkpoint.py` | Checks the checkpoint files and calculates evaluation metrics |

## Methods

Email text is represented using TF-IDF features, and logistic regression is used as the classifier. The data is divided into training, validation, and test sets. Cross-dataset experiments train on one dataset and evaluate on a different dataset.

Average Precision (AP) and Area Under the ROC Curve (AUROC) are used to evaluate model performance.

## Running the Project

Run these commands from the project directory after installing the required Python packages and obtaining the datasets:

```bash
python3 prepare_data.py
python3 make_split.py
python3 train_model.py
python3 cross_dataset.py
python3 plot_results.py
```

To check the checkpoint files, run:

```bash
python3 check_checkpoint.py --split split.csv --scores scores.csv --threshold 0.5
```

The original datasets and `prepared_data.csv` are not included in this repository because of their size. Obtain the datasets from their original sources before running the preparation and training scripts.

## Results

The cross-dataset experiments produced different results depending on which dataset was used for training and testing. The results are recorded in `cross_dataset_results.csv`, and the AUROC comparisons are shown in `cross_dataset_auroc.png`.

Very high scores should be interpreted cautiously because dataset-specific patterns or other artifacts may affect performance.

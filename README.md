# Phishing Email Detection Across Datasets

**Course:** COSC 4371 — Security Analytics  
**Project:** Project 5 — Final Checkpoint

## Overview

This project investigates how well a phishing email detection model generalizes across different email datasets. It uses TF-IDF to convert email text into numerical features and logistic regression to classify emails as legitimate or phishing/spam.

Three datasets are used:

- Phishing_Email
- CEAS-08
- IBM email dataset

The main goal is to evaluate cross-dataset generalization by training on one dataset and testing on another.

## Datasets

| Dataset | File |
|---|---|
| Phishing_Email | `Phishing_Email.csv` |
| CEAS-08 | `CEAS_08.csv` |
| IBM | `train-00000-of-00001.csv` and `test-00000-of-00001.csv` |

The original datasets and prepared data are not included in this repository because of their size. Obtain the datasets from their original sources and place the required files in the project directory.

## Project Files

| File | Description |
|---|---|
| `prepare_data.py` | Cleans and prepares the datasets |
| `check_duplicates.py` | Checks for duplicate emails |
| `check_overlap.py` | Checks for overlap between datasets |
| `make_split.py` | Creates training, validation, and test splits |
| `train_model.py` | Trains and evaluates models within each dataset |
| `cross_dataset.py` | Runs cross-dataset experiments |
| `cross_dataset_scores.csv` | Stores individual cross-dataset prediction scores |
| `cross_dataset_summary.csv` | Stores summarized cross-dataset metrics |
| `cross_dataset_results.csv` | Stores earlier experiment results |
| `cross_dataset_auroc.png` | Visualizes cross-dataset AUROC results |
| `check_final_checkpoint.py` | Validates final checkpoint files and calculates evaluation metrics |
| `check_near_duplicates.py` | Checks for near-duplicate email content |
| `near_duplicate_results.csv` | Stores near-duplicate analysis results |
| `near_duplicate_sensitivity.csv` | Stores results at different similarity thresholds |
| `overlap_sensitivity.py` | Tests sensitivity to removing overlapping records |
| `errors.csv` | Provides the error sample for checkpoint evaluation |
| `environment.txt` | Records the software environment |
| `split.csv` | Records dataset assignments, labels, and splits |

## Methods

### Data preparation

The datasets were cleaned and normalized before model training. Missing email text and duplicate records were addressed. Exact normalized-text overlap was checked across datasets and between data splits to reduce the risk of data leakage.

The label convention is:

- `0` = legitimate email
- `1` = phishing or spam email

The data was divided into training, validation, and test sets using stratified splitting.

### Model

The classifier uses a scikit-learn pipeline consisting of:

- TF-IDF vectorization
- Maximum features: 50,000
- N-gram range: (1, 2)
- Logistic regression
- Maximum iterations: 1,000

The decision threshold was selected using the validation set by maximizing F1 score. The selected threshold for each training dataset was then applied when evaluating that model on the other datasets.

### Cross-dataset evaluation

Three models were trained, one on each dataset. Each model was evaluated on the test sets of the other two datasets, producing six cross-dataset experiments.

Average Precision (AP) and Area Under the ROC Curve (AUROC) were calculated. AUROC is used as the primary comparison metric because the datasets have different phishing-class proportions.

## Cross-Dataset Results

The following table reports AUROC and its 95% bootstrap confidence interval.

| Training dataset | Test dataset | AUROC | 95% confidence interval |
|---|---|---:|---|
| Phishing_Email | CEAS-08 | 0.9444 | 0.9382–0.9504 |
| Phishing_Email | IBM | 0.9975 | 0.9962–0.9986 |
| CEAS-08 | Phishing_Email | 0.8086 | 0.7898–0.8250 |
| CEAS-08 | IBM | 0.8870 | 0.8705–0.9022 |
| IBM | Phishing_Email | 0.9171 | 0.9063–0.9281 |
| IBM | CEAS-08 | 0.9332 | 0.9257–0.9400 |

The results show that model performance varies depending on the training and testing datasets. This suggests that differences in dataset characteristics can affect how well a phishing detector generalizes.

Very high scores, especially the Phishing_Email-to-IBM result, should be interpreted cautiously because dataset-specific patterns or artifacts may influence performance.

## Leakage and Near-Duplicate Checks

Exact normalized-text overlap checks were performed across datasets and between training, validation, and test splits.

A near-duplicate sensitivity analysis was also conducted on the Phishing_Email-to-IBM comparison.

| Similarity threshold | Rows removed | Rows remaining | AUROC |
|---|---:|---:|---:|
| Original | 0 | 1,766 | 0.9975 |
| 0.90 | 218 | 1,548 | 0.9968 |
| 0.95 | 150 | 1,616 | 0.9970 |
| 0.99 | 24 | 1,742 | 0.9974 |

The AUROC remained very high after removing near-duplicate matches at the tested thresholds. This sensitivity check did not explain the high score, although dataset-specific artifacts and other forms of similarity may still affect the result.

## Running the Project

Run the commands from the project directory after installing the required Python packages and obtaining the datasets.

Prepare the data and create the splits:

```bash
python3 prepare_data.py
python3 make_split.py
```

Run the within-dataset and cross-dataset experiments:

```bash
python3 train_model.py
python3 cross_dataset.py
python3 plot_results.py
```

Run the final checkpoint evaluation:

```bash
python3 check_final_checkpoint.py --split split.csv --scores cross_dataset_scores.csv --errors errors.csv --metric auroc
```

The final checker reports evaluation metrics, confidence intervals, class counts, and error-sample information.

## Software Environment

The experiments were run on macOS with an Apple ARM64 processor.

- Python: 3.14.3
- scikit-learn: 1.9.1
- pandas: 3.0.5
- NumPy: 2.4.6
- SciPy: 1.18.1

See `environment.txt` for the recorded environment information.

## Repository

GitHub: https://github.com/yselim26/phishing-emails

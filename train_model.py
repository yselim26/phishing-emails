
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)

# Load the exact data used to create split.csv
data = pd.read_csv("prepared_data.csv", low_memory=False)

# Confirm the split labels match
split = pd.read_csv("split.csv")
check = data[["row_id", "dataset", "split", "label"]].merge(
    split,
    on=["row_id", "dataset", "split", "label"],
    how="outer",
    indicator=True
)
print("Row alignment check:")
print(check["_merge"].value_counts())

results = []

# Train and evaluate a separate baseline model for each dataset
for dataset_name in data["dataset"].unique():

    train = data[
        (data["dataset"] == dataset_name) &
        (data["split"] == "train")
    ].dropna(subset=["text", "label"])

    validation = data[
        (data["dataset"] == dataset_name) &
        (data["split"] == "validation")
    ].dropna(subset=["text", "label"])

    test = data[
        (data["dataset"] == dataset_name) &
        (data["split"] == "test")
    ].dropna(subset=["text", "label"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train only on training rows
    model.fit(train["text"].astype(str), train["label"])

    # Select threshold using validation F1
    val_scores = model.predict_proba(
        validation["text"].astype(str)
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        validation["label"], val_scores
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = np.argmax(f1)
    threshold = thresholds[best_index]

    print(
        dataset_name,
        "validation-selected threshold:",
        round(threshold, 4)
    )

    # Score test rows; don't use test labels to choose the threshold
    scores = model.predict_proba(
        test["text"].astype(str)
    )[:, 1]

    print(
        dataset_name,
        "AP:", round(average_precision_score(test["label"], scores), 4),
        "AUROC:", round(roc_auc_score(test["label"], scores), 4)
    )

    results.append(pd.DataFrame({
        "row_id": test["row_id"],
        "test_set": dataset_name,
        "label": test["label"],
        "score": scores
    }))

scores = pd.concat(results, ignore_index=True)
scores.to_csv("scores.csv", index=False)
print("\nCreated scores.csv with", len(scores), "rows.")

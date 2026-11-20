
import time
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)

# Load the exact prepared data used for the project splits
data = pd.read_csv("prepared_data.csv", low_memory=False)

datasets = ["Phishing_Email", "CEAS_08", "IBM"]

all_scores = []
summary = []
feature_rows = []

for source in datasets:

    # Get training and validation data from the source dataset
    train = data[
        (data["dataset"] == source) &
        (data["split"] == "train")
    ].dropna(subset=["text", "label"])

    validation = data[
        (data["dataset"] == source) &
        (data["split"] == "validation")
    ].dropna(subset=["text", "label"])

    # Build the model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train and measure training time
    start_time = time.time()
    model.fit(train["text"].astype(str), train["label"])
    training_time = time.time() - start_time

    # Extract logistic regression feature importance
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    top_positive = coefficients.argsort()[-20:][::-1]
    top_negative = coefficients.argsort()[:20]

    for index in top_positive:
        feature_rows.append({
            "training_dataset": source,
            "feature": feature_names[index],
            "coefficient": coefficients[index],
            "direction": "phishing"
        })

    for index in top_negative:
        feature_rows.append({
            "training_dataset": source,
            "feature": feature_names[index],
            "coefficient": coefficients[index],
            "direction": "legitimate"
        })
    # Choose threshold using source validation data only
    validation_scores = model.predict_proba(
        validation["text"].astype(str)
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        validation["label"], validation_scores
    )

    f1 = (
        2 * precision[:-1] * recall[:-1] /
        (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = f1.argmax()
    threshold = float(thresholds[best_index])

    print("\nTraining dataset:", source)
    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Selected threshold:", round(threshold, 4))
    print("Training time (seconds):", round(training_time, 2))

    # Test on each of the other datasets
    for target in datasets:

        if target == source:
            continue

        test = data[
            (data["dataset"] == target) &
            (data["split"] == "test")
        ].dropna(subset=["text", "label"])

        test_scores = model.predict_proba(
            test["text"].astype(str)
        )[:, 1]

        ap = average_precision_score(
            test["label"], test_scores
        )

        auroc = roc_auc_score(
            test["label"], test_scores
        )

        summary.append({
            "training_dataset": source,
            "test_dataset": target,
            "average_precision": ap,
            "auroc": auroc,
            "threshold": threshold,
            "training_seconds": training_time
        })

        # Save scores in the format required by the final checker
        pair_scores = pd.DataFrame({
            "row_id": test["row_id"],
            "value": source,
            "column": target,
            "label": test["label"].astype(int),
            "score": test_scores,
            "threshold": threshold
        })

        all_scores.append(pair_scores)

        print(
            source, "->", target,
            "| AP:", round(ap, 4),
            "| AUROC:", round(auroc, 4)
        )

# Save all six comparisons
scores_df = pd.concat(all_scores, ignore_index=True)
scores_df.to_csv("cross_dataset_scores.csv", index=False)

summary_df = pd.DataFrame(summary)
summary_df.to_csv("cross_dataset_summary.csv", index=False)

print("\nSaved cross_dataset_scores.csv")
print("Saved cross_dataset_summary.csv")
print("\nFinal comparison table:")
print(summary_df.to_string(index=False))
features_df = pd.DataFrame(feature_rows)
features_df.to_csv("feature_importance.csv", index=False)

print("\nSaved feature_importance.csv")
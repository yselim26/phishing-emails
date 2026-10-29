
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

# Load the prepared data used by the checkpoint
data = pd.read_csv("prepared_data.csv", low_memory=False)

# Train on Phishing_Email
train = data[
    (data["dataset"] == "Phishing_Email") &
    (data["split"] == "train")
].dropna(subset=["text", "label"])

# Test on IBM
test = data[
    (data["dataset"] == "IBM") &
    (data["split"] == "test")
].dropna(subset=["text", "label"])

print("Training dataset: Phishing_Email")
print("Training rows:", len(train))
print("Testing dataset: IBM")
print("Testing rows:", len(test))

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(train["text"].astype(str), train["label"])

scores = model.predict_proba(test["text"].astype(str))[:, 1]

print("\nCross-dataset results:")
print("Average Precision:", round(
    average_precision_score(test["label"], scores), 4
))
print("AUROC:", round(
    roc_auc_score(test["label"], scores), 4
))

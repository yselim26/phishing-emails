import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("prepared_data.csv", low_memory=False)

train = df[
    (df["dataset"] == "Phishing_Email") &
    (df["split"] == "train")
].copy()

test = df[
    (df["dataset"] == "IBM") &
    (df["split"] == "test")
].copy()

train["normalized_text"] = train["normalized_text"].fillna("").astype(str)
test["normalized_text"] = test["normalized_text"].fillna("").astype(str)

train = train[train["normalized_text"].str.len() > 0]
test = test[test["normalized_text"].str.len() > 0]

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

combined = pd.concat([
    train["normalized_text"],
    test["normalized_text"]
])

vectorizer.fit(combined)

X_train = vectorizer.transform(train["normalized_text"])
X_test = vectorizer.transform(test["normalized_text"])

neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute"
)

neighbors.fit(X_train)
distances, indices = neighbors.kneighbors(X_test)

results = pd.DataFrame({
    "ibm_row_id": test["row_id"].to_numpy(),
    "ibm_label": test["label"].to_numpy(),
    "phishing_email_row_id": train.iloc[indices.flatten()]["row_id"].to_numpy(),
    "phishing_email_label": train.iloc[indices.flatten()]["label"].to_numpy(),
    "similarity": 1 - distances.flatten()
})

results.to_csv("near_duplicate_results.csv", index=False)

print("Compared IBM test emails:", len(test))
print("Compared Phishing_Email training emails:", len(train))

for threshold in [0.80, 0.90, 0.95, 0.99]:
    print(
        f"Similarity >= {threshold:.2f}:",
        (results["similarity"] >= threshold).sum()
    )

print("\nHighest similarity matches:")
print(results.sort_values("similarity", ascending=False).head(10).to_string(index=False))

print("\nSaved near_duplicate_results.csv")

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

scores = pd.read_csv("cross_dataset_scores.csv")
near = pd.read_csv("near_duplicate_results.csv")

print("IBM test sensitivity analysis")
print("Original IBM test rows:", len(near))

for threshold in [None, 0.80, 0.90, 0.95, 0.99]:
    if threshold is None:
        filtered = scores[scores["column"] == "IBM"].copy()
        name = "All IBM test emails"
    else:
        excluded = set(
            near.loc[near["similarity"] >= threshold, "ibm_row_id"]
        )
        filtered = scores[
            (scores["column"] == "IBM") &
            (~scores["row_id"].isin(excluded))
        ].copy()
        name = f"Similarity below {threshold:.2f}"

    print(f"\n{name}")
    print("Rows remaining:", len(filtered))

    for training_set, group in filtered.groupby("value"):
        ap = average_precision_score(group["label"], group["score"])
        auc = roc_auc_score(group["label"], group["score"])
        print(f"{training_set}: AP={ap:.4f}, AUROC={auc:.4f}")

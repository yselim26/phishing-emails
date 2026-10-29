
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("cross_dataset_results.csv")

results["comparison"] = (
    results["training_dataset"] + " → " +
    results["test_dataset"]
)

plt.figure(figsize=(10, 5))
plt.bar(results["comparison"], results["auroc"])
plt.ylabel("AUROC")
plt.xlabel("Training → Test dataset")
plt.title("Cross-Dataset Phishing Detection Performance")
plt.ylim(0, 1)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("cross_dataset_auroc.png", dpi=200)
plt.show()

print("Saved cross_dataset_auroc.png")

import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("cross_dataset_results.csv")

results["comparison"] = (
    results["training_dataset"] + " → " +
    results["test_dataset"]
)

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(results["comparison"], results["auroc"])

ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=10)

ax.set_ylabel("AUROC", fontsize=12)
ax.set_xlabel("Training → Test dataset", fontsize=12)
ax.set_title("Cross-Dataset Phishing Detection Performance", fontsize=15)

ax.set_ylim(0, 1.06)
ax.tick_params(axis="x", labelrotation=35)
plt.setp(ax.get_xticklabels(), ha="right")

plt.tight_layout()
plt.savefig("cross_dataset_auroc.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved updated cross_dataset_auroc.png")

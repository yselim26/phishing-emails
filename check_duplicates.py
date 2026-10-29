import pandas as pd

datasets = {
    "Phishing Email": {
        "file": "Phishing_Email.csv",
        "text": "Email Text",
        "label": "Email Type"
    },
    "IBM train": {
        "file": "train-00000-of-00001.csv",
        "text": "features",
        "label": "target"
    },
    "IBM test": {
        "file": "test-00000-of-00001.csv",
        "text": "features",
        "label": "target"
    },
    "CEAS-08": {
        "file": "CEAS_08.csv",
        "text": "body",
        "label": "label"
    }
}

for name, info in datasets.items():
    df = pd.read_csv(info["file"])

    # Count exact duplicate rows
    exact_duplicates = df.duplicated().sum()

    # Normalize text for comparison
    text = (
        df[info["text"]]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.split()
        .str.join(" ")
    )

    # Count repeated email text
    repeated_rows = text.duplicated().sum()
    repeated_groups = (text.value_counts() > 1).sum()

    # Check whether identical text has different labels
    temp = pd.DataFrame({
        "text": text,
        "label": df[info["label"]]
    })

    label_counts = temp.groupby("text")["label"].nunique()
    conflicting_groups = (label_counts > 1).sum()

    print("\nDataset:", name)
    print("Exact duplicate rows:", exact_duplicates)
    print("Rows with repeated text:", repeated_rows)
    print("Groups of repeated text:", repeated_groups)
    print("Repeated-text groups with conflicting labels:",
          conflicting_groups)
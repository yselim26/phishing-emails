
import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
d1 = pd.read_csv("Phishing_Email.csv")
ibm_train = pd.read_csv("train-00000-of-00001.csv")
ibm_test = pd.read_csv("test-00000-of-00001.csv")
ceas = pd.read_csv("CEAS_08.csv")

# Standardize Dataset 1
d1 = d1.rename(columns={
    "Email Text": "text",
    "Email Type": "original_label"
})
d1["label"] = d1["original_label"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})
d1["dataset"] = "Phishing_Email"
d1 = d1.dropna(subset=["text"]).copy()

# Standardize CEAS-08
ceas = ceas.rename(columns={"body": "text"})
ceas["dataset"] = "CEAS_08"
ceas = ceas.dropna(subset=["text"]).copy()

# Combine IBM train and test
ibm = pd.concat([ibm_train, ibm_test], ignore_index=True)
ibm = ibm.rename(columns={
    "features": "text",
    "target": "label"
})
ibm["dataset"] = "IBM"
ibm = ibm.dropna(subset=["text"]).copy()

# Normalize email text for duplicate checking
def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.split()
        .str.join(" ")
    )

d1["normalized_text"] = normalize_text(d1["text"])
ceas["normalized_text"] = normalize_text(ceas["text"])
ibm["normalized_text"] = normalize_text(ibm["text"])

# Remove duplicates within each dataset
d1 = d1.drop_duplicates(subset="normalized_text").copy()
ceas = ceas.drop_duplicates(subset="normalized_text").copy()
ibm = ibm.drop_duplicates(subset="normalized_text").copy()

# Remove IBM emails that appear in either other dataset
existing_texts = set(d1["normalized_text"]) | set(ceas["normalized_text"])
ibm = ibm[~ibm["normalized_text"].isin(existing_texts)].copy()

# Split each dataset into train, validation, and test
def split_dataset(df):
    train, remaining = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=42,
        stratify=remaining["label"]
    )

    train = train.assign(split="train")
    validation = validation.assign(split="validation")
    test = test.assign(split="test")

    return pd.concat([train, validation, test])

combined = pd.concat([
    split_dataset(d1),
    split_dataset(ceas),
    split_dataset(ibm)
], ignore_index=True)

# Create unique row IDs
combined = combined.reset_index(drop=True)
combined["row_id"] = [
    f"row_{i:06d}" for i in range(len(combined))
]

# Save the complete prepared data, including email text
combined.to_csv("prepared_data.csv", index=False)

# Save the required split file
split_file = combined[["row_id", "dataset", "split", "label"]]
split_file.to_csv("split.csv", index=False)

print("Created prepared_data.csv and split.csv")

print("Created split.csv")
print("\nRows per dataset, split, and label:")
print(combined.groupby(["dataset", "split", "label"]).size())
print("\nTotal rows:", len(combined))

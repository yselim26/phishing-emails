
import pandas as pd

# Load the datasets
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

# Standardize IBM column names
ibm = ibm.rename(columns={
    "features": "text",
    "target": "label"
})

ibm["dataset"] = "IBM"
ibm = ibm.dropna(subset=["text"]).copy()

print("\nIBM combined:", len(ibm), "rows")
print(ibm["label"].value_counts().sort_index())

# Display current counts
print("Dataset 1:", len(d1), "rows")
print(d1["label"].value_counts().sort_index())

print("\nCEAS-08:", len(ceas), "rows")
print(ceas["label"].value_counts().sort_index())

print("\nIBM train:", len(ibm_train), "rows")
print(ibm_train["target"].value_counts().sort_index())

print("\nIBM test:", len(ibm_test), "rows")
print(ibm_test["target"].value_counts().sort_index())



# Normalize email text for duplicate checking
def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.split()
        .str.join(" ")
    )
# Normalize email text in all three datasets
d1["normalized_text"] = normalize_text(d1["text"])
ceas["normalized_text"] = normalize_text(ceas["text"])
ibm["normalized_text"] = normalize_text(ibm["text"])

# Remove duplicate emails within each dataset
d1_before = len(d1)
ceas_before = len(ceas)
ibm_before = len(ibm)

d1 = d1.drop_duplicates(subset="normalized_text").copy()
ceas = ceas.drop_duplicates(subset="normalized_text").copy()
ibm = ibm.drop_duplicates(subset="normalized_text").copy()

# Find IBM emails already present in Dataset 1 or CEAS-08
existing_texts = set(d1["normalized_text"]) | set(ceas["normalized_text"])

overlap = ibm["normalized_text"].isin(existing_texts).sum()
ibm = ibm[~ibm["normalized_text"].isin(existing_texts)].copy()

print("\nDuplicate cleanup:")
print("Dataset 1 duplicates removed:", d1_before - len(d1))
print("CEAS-08 duplicates removed:", ceas_before - len(ceas))
print("IBM duplicates removed:", ibm_before - len(ibm) - overlap)
print("IBM emails overlapping other datasets:", overlap)
print("IBM rows remaining:", len(ibm))

# Remove duplicate emails within each dataset
d1["normalized_text"] = normalize_text(d1["text"])
ceas["normalized_text"] = normalize_text(ceas["text"])

d1_before = len(d1)
ceas_before = len(ceas)

d1 = d1.drop_duplicates(subset="normalized_text", keep="first").copy()
ceas = ceas.drop_duplicates(subset="normalized_text", keep="first").copy()

print("\nDuplicates removed within each dataset:")
print("Dataset 1:", d1_before - len(d1))
print("CEAS-08:", ceas_before - len(ceas))

# Check overlap between the cleaned datasets
shared_texts = set(d1["normalized_text"]) & set(ceas["normalized_text"])
print("\nShared emails remaining between Dataset 1 and CEAS-08:", len(shared_texts))


# Check whether repeated email text has conflicting labels
for name, df in [("Dataset 1", d1), ("CEAS-08", ceas)]:
    conflicts = (
        df.groupby("normalized_text")["label"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"{name} conflicting-label groups: {conflicts}")

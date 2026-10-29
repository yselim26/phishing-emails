
import pandas as pd

# Load the email text from each dataset
d1 = pd.read_csv("Phishing_Email.csv")["Email Text"]
ibm_train = pd.read_csv("train-00000-of-00001.csv")["features"]
ibm_test = pd.read_csv("test-00000-of-00001.csv")["features"]
ceas = pd.read_csv("CEAS_08.csv")["body"]

def normalize(series):
    return (
        series.dropna()
        .astype(str)
        .str.lower()
        .str.split()
        .str.join(" ")
    )

# Normalize text for comparison
a = normalize(d1)
b = normalize(ibm_train)
c = normalize(ibm_test)
d = normalize(ceas)

# Find IBM emails that also appear in Dataset 1
d1_texts = set(a)

train_overlap = b.isin(d1_texts)
test_overlap = c.isin(d1_texts)

print("IBM train rows overlapping Dataset 1:", train_overlap.sum())
print("IBM test rows overlapping Dataset 1:", test_overlap.sum())

# Check overlap between the original IBM train and test splits
print("IBM train/test shared texts:", len(set(b) & set(c)))

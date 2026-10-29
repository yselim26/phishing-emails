import pandas as pd

# The four CSV files I am using in my project
files = [
    "Phishing_Email(2).csv",
    "train-00000-of-00001(2).csv",
    "test-00000-of-00001(2).csv",
    "CEAS_08(1).csv"
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)

    # Load the CSV file
    df = pd.read_csv(file)

    print("Rows and columns:", df.shape)

    print("Columns:", df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    # Check the labels, if a label column exists
    for column in ["Email Type", "target", "label"]:
        if column in df.columns:
            print(f"\nLabel counts for '{column}':")
            print(df[column].value_counts(dropna=False))

    print("\nFirst two rows:")
    print(df.head(2).to_string(index=False))
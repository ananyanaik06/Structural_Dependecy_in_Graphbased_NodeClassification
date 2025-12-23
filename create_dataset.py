import pandas as pd

FEATURES_FILE = "data/graph_features.csv"
LABELS_FILE = "data/node_labels.csv"
OUTPUT_FILE = "data/node_dataset.csv"

features_df = pd.read_csv(FEATURES_FILE)
labels_df = pd.read_csv(LABELS_FILE)

dataset_df = pd.merge(
    features_df,
    labels_df,
    on="node_id",
    how="inner"   # keep only nodes with labels
)

print(dataset_df.head(10)) # print the first 5 rows

dataset_df.to_csv(OUTPUT_FILE, index=False)

print("Merged dataset shape:", dataset_df.shape)

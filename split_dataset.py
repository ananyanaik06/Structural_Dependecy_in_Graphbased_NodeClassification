import pandas as pd
import numpy as np

df = pd.read_csv("data/node_dataset.csv")

#usually training is 80% ( but not necessary)
train_frac = 0.8
random_seed = 42 # to ensure on every run you get the same shuffled data frame
np.random.seed(random_seed)

train_rows = []
test_rows = []

for label in df["label"].unique():
    class_df = df[df["label"] == label]
    
    shuffled = class_df.sample(frac=1, random_state=random_seed)
    
    split_idx = int(len(shuffled) * train_frac)
    
    train_rows.append(shuffled.iloc[:split_idx]) # iloc means integer location
    test_rows.append(shuffled.iloc[split_idx:])

train_df = pd.concat(train_rows).sample(frac=1, random_state=random_seed)
test_df = pd.concat(test_rows).sample(frac=1, random_state=random_seed)

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

print("\nTrain label distribution:")
print(train_df["label"].value_counts(normalize=True)) # returns the fraction of each label 

print("\nTest label distribution:")
print(test_df["label"].value_counts(normalize=True)) # returns the fraction of each label 

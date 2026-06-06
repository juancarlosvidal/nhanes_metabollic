"""
Generate and save train/test and CV fold splits based on SEQN (patient ID).
Run this once before executing any of the comparison notebooks.
Splits are stratified on metabolic_syndrome and saved to data/splits.npz.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

RANDOM_STATE = 0
N_FOLDS = 5
FILE = './data/nhanes_with_metabolic_syndrome_adults.csv'
TARGET = 'metabolic_syndrome'
WEIGHTS = 'WTMEC2YR'
OUTPUT = './data/splits.npz'

df = pd.read_csv(FILE, index_col=0, low_memory=False)
df = df.dropna(subset=[TARGET, WEIGHTS])

seqn = df.index.to_numpy()
y = df[TARGET].to_numpy()

# Stratified 80/20 train/test split
train_seqn, test_seqn, y_train, _ = train_test_split(
    seqn, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# Stratified K-Fold on the training set
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_data = {}
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_seqn, y_train)):
    fold_data[f'fold_{fold_idx}_train'] = train_seqn[train_idx]
    fold_data[f'fold_{fold_idx}_val'] = train_seqn[val_idx]

np.savez(OUTPUT, train_seqn=train_seqn, test_seqn=test_seqn, **fold_data)

print(f"Splits saved to {OUTPUT}")
print(f"  Total patients : {len(seqn)}")
print(f"  Train          : {len(train_seqn)}  (pos rate: {y_train.mean():.3f})")
print(f"  Test           : {len(test_seqn)}")
for fold_idx in range(N_FOLDS):
    nt = len(fold_data[f'fold_{fold_idx}_train'])
    nv = len(fold_data[f'fold_{fold_idx}_val'])
    print(f"  Fold {fold_idx}         : train={nt}, val={nv}")

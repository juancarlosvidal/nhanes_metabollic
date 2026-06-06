# Core dataset utilities: PyTorch Dataset wrapper and feature normalization helpers.
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight


class CustomDataset(Dataset):
    """PyTorch Dataset that wraps feature matrix x, label vector y, and sample weights w."""

    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors for a single sample
        xt = torch.from_numpy(self.x[idx, :])
        yt = torch.from_numpy(self.y[idx])
        wt = torch.from_numpy(self.w[idx])
        return xt, yt, wt


def num_to_onehot(sample):
    """Convert a 1-D integer array into a one-hot encoded matrix (n_samples x n_categories)."""
    # Get unique values
    if len(sample.shape) > 1:
        sample = np.squeeze(sample)
    categories = np.sort(np.unique(sample))
    new_array = np.zeros((sample.shape[0], len(categories)))
    for s in range(len(categories)):
        index = sample == categories[s]
        new_array[index, s] = 1
    return new_array


def normalize_column(column):
    """Min-max normalize a pandas Series or array to the [0, 1] range."""
    min_c = column.min()
    max_c = column.max()
    eps = np.finfo(float).eps
    feature_vector = (column.values - min_c) / (max_c - min_c + eps)
    return feature_vector


def normalize_value(value, column):
    """
    Normalize a single value using the min/max of a reference column.

    Parameters:
    -----------
    value : float
        Original value to normalize.
    column : pd.Series or np.array
        Reference column used to derive min and max.

    Returns:
    --------
    float : normalized value in the [0, 1] range.
    """
    min_c = column.min()
    max_c = column.max()
    eps = np.finfo(float).eps
    normalized = (value - min_c) / (max_c - min_c + eps)
    return normalized


def denormalize_value(normalized_value, column):
    """
    Reverse min-max normalization for a single value.

    Parameters:
    -----------
    normalized_value : float
        Normalized value (expected in [0, 1]).
    column : pd.Series or np.array
        Reference column used to derive min and max.

    Returns:
    --------
    float : original (de-normalized) value.
    """
    min_c = column.min()
    max_c = column.max()
    eps = np.finfo(float).eps
    original = normalized_value * (max_c - min_c + eps) + min_c
    return original

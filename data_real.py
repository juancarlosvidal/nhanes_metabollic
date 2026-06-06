# Data loader for real NHANES datasets, with separate handling for continuous and categorical variables.
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

from data import normalize_column, num_to_onehot


def load_data(file, variables, output_var, weights_var):
    """
    Load a real-data CSV file (NHANES format) and return a data dict for model training.

    Continuous variables are min-max normalized; categorical variables are one-hot encoded.

    Parameters:
    -----------
    file : str
        Path to the CSV file.
    variables : list of str
        Subset of supported NHANES variable codes to use as features.
    output_var : str
        Name of the target column (e.g., 'diabetes').

    Returns:
    --------
    dict with keys: 'seqn', 'x', 'x_norm', 'y', 'w', 'z', 'sigma', 'colnames'
    """
    # Load csv
    df = pd.read_csv(file, index_col=0, low_memory=False)

    # # Drop rows with NaNs in non-feature columns (target and weights must be complete)
    # df = df.dropna(subset=[output_var, weights_var])
    # # Impute NaNs in feature columns with the column median
    # df[variables] = df[variables].fillna(df[variables].median())

    # Drop rows with NaNs in non-feature columns (target and weights must be complete)
    df = df.dropna(subset=variables + [output_var, weights_var])

    L = []       # Raw (unnormalized) features
    L_norm = []  # Normalized features
    colnames = []

    # Supported continuous (integer-scale) NHANES variables
    int_variables = [
        "INQ020_7",
        "INDFMMPI_7",
        "WHD050_30",
        "alcoholfrecuencia",
        "RIDAGEYR",
        "BMXHT",
        "BMXWT",
        "BMXBMI",
        "BMXWAIST",
        "LBXTR_64",
        "BPXDI1",
        "BPXSY1",
        "BPXPLS",
        "BPXDI1",
        "LBDSCHSI_43",
        "LBXSTR_43",
        "LBXSGL_43",
        "LBXGH_39",
        "LBXSCH",
        "LBXSTR",
        "LBXGH",
        "LBXGLU",
        "LBXGH",
    ]

    # Supported categorical NHANES variables (will be one-hot encoded)
    cat_variables = [
        "BPQ020_40",
        "HIQ011_1",
        "HOQ065_13",
        "KIQ026_19",
        "MCQ160K_35",
        "MCQ160N_35",
        "MCQ220_35",
        "MCQ365A_35",
        "MCQ365D_35",
        "MCQ365B_35",
        "SLQ050_9",
        "MCQ220",
        "RIAGENDR",
        "RIDRETH3",
    ]

    all_variables = int_variables + cat_variables

    for i, c in enumerate(variables):
        assert c in all_variables, "Variable {} not included".format(c)

        if c in int_variables:
            feature_vector = df[c].values
            L.extend([np.expand_dims(feature_vector, 1)])
            # Normalize to 0-1 range
            feature_vector = normalize_column(df[c])
            L_norm.extend([np.expand_dims(feature_vector, 1)])
            colnames.extend([c])

        elif c in cat_variables:
            # One-hot encoding for categorical variables
            feature_vector = num_to_onehot(df[c].to_numpy())
            L.extend([feature_vector])
            L_norm.extend([feature_vector])
            for j in range(feature_vector.shape[1]):
                colnames.extend([c + "_" + str(j)])

    seqn = df.index.to_numpy()
    x = np.hstack(L).astype("float32")
    x_norm = np.hstack(L_norm).astype("float32")
    y = df[output_var].values.astype("float32").reshape(-1, 1)  # nn.BCELoss()
    # Inverse-probability weights normalized to sum to 1
    w = 1 / df[weights_var].values.astype("float32").reshape(-1, 1)
    w = w / np.sum(w)
    
    # Random noise term for conformal inference
    # z = np.random.uniform(0, 1, len(y)).reshape(-1, 1)
    z = np.random.standard_normal(len(y)).reshape(-1, 1)
    sigma = 0.01  # Noise scale

    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    data = {
        "seqn": seqn,
        "x": x,
        "x_norm": x_norm,
        "y": y,
        "w": w,
        "z": z,
        "sigma": sigma,
        "colnames": colnames,
    }

    return data

import numpy as np
import torch
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def introduce_missing(X):
    N, D = X.shape
    Xnan = X.copy()

    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz

def load_data_from_uci(id=17):

    # Fetch dataset
    dataset = fetch_ucirepo(id=id)

    # Data (as pandas dataframes)
    X_df = dataset.data.features
    y_df = dataset.data.targets  # Not used for this unsupervised task

    # Convert to numpy arrays
    X = X_df.values
    y = y_df.values

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def split_data(X, y, flag_val=False, test_size=0.2, random_state=42):
    # Split data into train and validation sets
    if flag_val: #make train and val different
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_val, y_train, y_val
    # We use as train and val data (without adding missing) the entire dataset
    X_train = X.copy()
    y_train = y.copy()
    X_val = X.copy()
    y_val = y.copy()
    return X_train, X_val, y_train, y_val

def preprocess_data(X, y, flag_val=False, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = split_data(X, y, flag_val, test_size, random_state)
    # introduce missing
    X_train_nan, X_train_missing = introduce_missing(X_train)
    X_val_nan, X_val_missing = introduce_missing(X_val)

    # create masks
    mask_train = np.array(~np.isnan(X_train_nan), dtype=np.float32)
    mask_val = np.array(~np.isnan(X_val_nan), dtype=np.float32)

    # convert to pytorch tensors
    X_train_nan = torch.tensor(X_train_nan, dtype=torch.float32)
    X_train_missing = torch.tensor(X_train_missing, dtype=torch.float32)
    X_val_nan = torch.tensor(X_val_nan, dtype=torch.float32)
    X_val_missing = torch.tensor(X_val_missing, dtype=torch.float32)
    mask_train = torch.tensor(mask_train, dtype=torch.float32)
    mask_val = torch.tensor(mask_val, dtype=torch.float32)


    return X_train_missing, mask_train, X_val_missing, mask_val, y_train, y_val

def load_and_preprocess_data(id=17, flag_val=False, test_size=0.2, random_state=42):
    X, y = load_data_from_uci(id)
    return X, y, preprocess_data(X, y, flag_val, test_size, random_state)
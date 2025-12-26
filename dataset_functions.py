import pandas as pd
import numpy as np

# Load dataset

def load_data(path="resources/dataset-uci.csv"):
    df = pd.read_csv(path, header=0)
    data = df.to_numpy()
    y = data[:, 0].astype(int)
    y = np.where(y == 0, -1, 1)
    X_features = data[:, 1:].astype(float)

    return X_features, y


def normaliser_fit(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0

    return mean, std


def normaliser_apply(X, mean, std):
    return (X - mean) / std


def split_train_test(X, y, test_ratio=0.2, shuffle=True, seed=None):
    """
    Used for cross validation
    """

    if seed is not None:
        np.random.seed(seed)

    dataset_size = X.shape[0]
    indices = np.arange(dataset_size)

    if shuffle:
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    split_point = int( (1.0 - test_ratio) * dataset_size)
    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]

    return X_train, y_train, X_test, y_test






# preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

def load_data(path):
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)

def clean_data(df, is_train=True):
    """
    Fills missing values and drops unused columns.
    is_train=True means we're working with training data (which has a target and PassengerId).
    """
    # Fill missing numerical values with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Fill missing categorical values with mode (most common)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Drop columns not useful for ML
    drop_cols = ['Name', 'Ticket', 'Cabin']
    if is_train:
        drop_cols.append('PassengerId')  # Only drop PassengerId from training set

    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df

def encode_features(df):
    """
    Label-encodes all categorical columns.
    Converts strings like 'male'/'female' into integers 0/1.
    """
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def scale_features(X):
    """
    Standardizes features to have 0 mean and 1 standard deviation.
    Helps neural network training converge faster.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def preprocess_training(path):
    """
    Full preprocessing pipeline for training data:
    - load
    - clean
    - encode
    - scale
    Returns: X (features), y (target)
    """
    df = load_data(path)
    df = clean_data(df, is_train=True)
    df = encode_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].values.reshape(-1, 1)

    X_scaled = scale_features(X)
    return X_scaled, y

def preprocess_test(path):
    """
    Full preprocessing pipeline for test data (without target column).
    Used for inference/submission.
    """
    df = load_data(path)
    df = clean_data(df, is_train=False)
    df = encode_features(df)
    X_scaled = scale_features(df)
    return X_scaled


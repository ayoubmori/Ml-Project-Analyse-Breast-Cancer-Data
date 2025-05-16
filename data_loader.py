import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df

def get_features_and_target(df):
    y = df['diagnosis']
    x = df.drop(['diagnosis'], axis=1)
    return x, y

def handle_outliers(x):
    processed_x = x.copy()
    for col in processed_x.columns:
        q1 = processed_x[col].quantile(0.25)
        q3 = processed_x[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        median = processed_x[col].mean()
        processed_x[col] = np.where((processed_x[col] < lower) | (processed_x[col] > upper), median, processed_x[col])
    return processed_x

def get_high_correlation_features(x, threshold=0.2):
    corr_matrix = x.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr = (corr_matrix > threshold) & mask
    cols = set(corr_matrix.columns[high_corr.any(axis=0)]).union(set(corr_matrix.columns[high_corr.any(axis=1)]))
    return x[list(cols)]
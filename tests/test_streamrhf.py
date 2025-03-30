import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve, auc
import configparser
import rhf_stream as rhfs  # Ensure this module is installed and accessible

# Load dataset
def load_dataset(dataset, data_path='./data/public/', shuffled=False):
    file_path = f"{data_path.strip('/')}/{dataset}.gz"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    if shuffled:
        df = df.sample(frac=1)
    
    labels = df['label'].to_numpy(dtype='float32')
    data = df.drop('label', axis=1).to_numpy(dtype='float32')
    return data, labels

# Read config
def read_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# Main function
def run_rhf_stream(data, labels, T=100, H=5, N_init_pts=1, iterations=1, shuffled=False):
    results = []
    data = np.array(data, dtype='float64')
    data = data.copy(order='C')

    for m in range(iterations):
        t0 = time.time()
        scores = rhfs.rhf_stream(data, T, H, N_init_pts)
        print('first score ' + str(scores[0]))
        t1 = time.time()

        # Calculate metrics
        ap_score = average_precision_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)

        results.append({
            "iteration": m,
            "ap_score": ap_score,
            "auc_score": auc_score,
            "time": t1 - t0,
        })

        # Reload data if shuffled
        if shuffled:
            data, labels = load_dataset("abalone", "./data/public/", shuffled=True)
            data = np.array(data, dtype='float64')
            data = data.copy(order='C')

    return results

# Wrapper for pytest
def process_dataset(fname="abalone", config_path="config.ini", iterations=1):
    config = read_config(config_path)
    data_path = config['DATA']['dataset_path']

    data, labels = load_dataset(fname, data_path, shuffled=True)
    N_init_pts = int(round(data.shape[0] * 0.01))  # Example: 1% of the dataset
    print("Number of initial points / window size: ", N_init_pts)
    return run_rhf_stream(data, labels, T=100, H=5, N_init_pts=N_init_pts, iterations=iterations)

import pytest

def test_process_dataset():
    # Run the processing function
    results = process_dataset(fname="abalone", config_path="config.ini", iterations=1)

    # Check output structure
    assert isinstance(results, list), "Results should be a list"
    assert len(results) == 1, "Should process only one iteration"
    assert "ap_score" in results[0], "AP score missing in results"
    print("AP Score:", results[0]["ap_score"])
    assert "auc_score" in results[0], "AUC score missing in results"
    print("AUC Score:", results[0]["auc_score"])
    assert results[0]["ap_score"] > 0, "AP score should be greater than 0"

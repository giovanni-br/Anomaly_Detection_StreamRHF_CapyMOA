import random_histogram_forest as rhfs
import numpy as np
from scipy.stats import kurtosis
import random
import time
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
import pandas as pd
from scipy.stats import sem

print("Starting...")

np.random.seed(42)

dataset_name = "thyroid"
#path = f"C:/Users/giova/Downloads/wetransfer_forstefan_2024-12-13_1400/forStefan/data/public/{dataset_name}.gz"
path = f"C:/Users/aleja/OneDrive - Universidad Nacional de Colombia/Documentos/Institut Polytechnique de Paris/courses/P1/Data Streaming/project/actual code/datasets/forStefan/data/public/{dataset_name}.gz"
df = pd.read_csv(path)

labels = df['label'].to_numpy(dtype='float32')
data_stream = df.drop('label', axis=1).to_numpy(dtype='float32')
print("data: ", data_stream.shape)

shuffled_indices = np.random.permutation(len(data_stream))
data_stream = data_stream[shuffled_indices]
labels = labels[shuffled_indices]

# Parameters
max_height = 5
num_trees = 100
window_size = len(data_stream) // 100  # 1% of the data stream
print("Window size: ", window_size)

n_runs = 1  # number of runs
ap_scores = []
execution_times = []

for i in range(n_runs):
    # Measure Execution Time
    start_time = time.time()

    # Run STREAMRHF
    anomaly_scores = rhfs.STREAMRHF(data_stream, max_height, num_trees, window_size, data_stream.shape[1])

    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

    # Calculate Average Precision Score
    ap_score = average_precision_score(labels, anomaly_scores)
    auc_score = roc_auc_score(labels, anomaly_scores)
    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
    print(f"Run {i + 1}: AP = {ap_score:.4f}, Execution Time = {execution_time:.2f} seconds")
    print(f"AUC Score from sklearn: {auc_score:.4f}")
    print("AUC Score from paper:", auc(fpr, tpr))
    ap_scores.append(ap_score)

# Convert to arrays
ap_scores = np.array(ap_scores)
execution_times = np.array(execution_times)

# Compute means
mean_ap = np.mean(ap_scores)
mean_time = np.mean(execution_times)

# Compute standard errors
ap_sem = sem(ap_scores)
time_sem = sem(execution_times)

# 95% Confidence interval = mean ± 1.96 * SEM
confidence_level = 1.96
ap_ci = confidence_level * ap_sem
time_ci = confidence_level * time_sem

# Print Results
print(f"Over {n_runs} runs:")
print(f"AP: {mean_ap:.4f} ± {ap_ci:.4f} (95% CI)")
print(f"Execution Time: {mean_time:.2f} ± {time_ci:.2f} seconds (95% CI)")
import numpy as np
from scipy.stats import kurtosis
import random
import time
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
import pandas as pd
from scipy.stats import sem

# Random Histogram Tree Node
class Node:
    def __init__(self, data, height, max_height, seed=None):
        self.data = data  # Data at this node
        self.height = height  # Current depth of the tree
        self.max_height = max_height  # Maximum allowed height of the tree
        self.seed = seed  # Random seed for attribute selection
        self.attribute = None  # Split attribute
        self.value = None  # Split value
        self.left = None  # Left child
        self.right = None  # Right child
    
    def is_leaf(self):
        return self.left is None and self.right is None

# Function to compute the kurtosis score as done in the RHF paper
#problem is that sometimes variance may be zero, so we need to add a small value to it
def compute_kurtosis(data):
    """
    Compute kurtosis for each feature (column) of the dataset and apply the log transformation:
    log[K(Xa) + 1], where K(Xa) is the kurtosis of feature Xa.

    Args:
    - data: The dataset (2D array), where each column is a feature

    Returns:
    - kurtosis_values: An array of log-transformed kurtosis values for each feature (column)
    """
    #print('data shape in compute_kurtosis: ', data.shape)
    # Ensure the data is in the correct format (2D array: instances x features)
    data = np.asarray(data)

    # Initialize an array to store kurtosis values for each feature
    kurtosis_values = np.zeros(data.shape[1])

    # Calculate the kurtosis for each feature (i.e., for each column)
    for feature_idx in range(data.shape[1]):
        # Extract the feature (column)
        feature_data = data[:, feature_idx]
        
        # Step 1: Calculate the mean (μ) for this feature
        mean = np.mean(feature_data)
        
        # Step 2: Calculate the second central moment (variance, σ^2)
        variance = np.mean((feature_data - mean) ** 2)
        
        # Step 3: Calculate the fourth central moment (μ4)
        fourth_moment = np.mean((feature_data - mean) ** 4)
        
        # Step 4: Calculate kurtosis for this feature (K(Xa))
        #print(variance)
        kurtosis_value = fourth_moment / ((variance + 1e-10) ** 2)
        
        # Step 5: Apply the log transformation: log[K(Xa) + 1]
        kurtosis_values[feature_idx] = np.log(kurtosis_value + 1)

    #print('output shape in compute_kurtosis: ', kurtosis_values.shape)
    #print('kurtosis values ' + str(kurtosis_values))
    #print('for the input data' + str(data))
    return kurtosis_values

# Function to randomly choose a splitting attribute based on kurtosis
def choose_split_attribute(kurt_values, random_seed):
    np.random.seed(random_seed)
    Ks = np.sum(kurt_values)
    #print('the sum of the kurtosis is ' + str(Ks))
    r = np.random.uniform(0, Ks)
    cumulative = 0
    for idx, k_value in enumerate(kurt_values):
        cumulative += k_value
        if cumulative > r:
            return idx
    return len(kurt_values) - 1

# Random Histogram Tree (RHT) Build Function, builds ONE tree
def RHT_build(data, height, max_height, seed_array):
    #print('type of data in RHT_build ' + str(type(data)))
    node = Node(data, height, max_height, seed_array[height])
    if height == max_height or len(data) <= 1:
        return node  # Return leaf node if max height is reached or data is small
    #print('calling kurtosis from RHT_build')
    kurt_values = compute_kurtosis(data)
    attribute = choose_split_attribute(kurt_values, node.seed)
    split_value = np.random.uniform(np.min(data[:, attribute]), np.max(data[:, attribute]))
    
    node.attribute = attribute
    node.value = split_value

    left_data = data[data[:, attribute] <= split_value]
    right_data = data[data[:, attribute] > split_value]

    node.left = RHT_build(left_data, height + 1, max_height, seed_array)
    node.right = RHT_build(right_data, height + 1, max_height, seed_array)
    return node

# Insert New Instance into Tree (Algorithm 1)
def insert(node, instance, max_height, seed_array):
    if not node.is_leaf():
        #print('calling kurtosis from insert')
        kurt_values = compute_kurtosis(np.vstack((node.data, instance)))
        new_attribute = choose_split_attribute(kurt_values, seed_array[node.height])
        if node.attribute != new_attribute:
            return RHT_build(np.vstack((node.data, instance)), node.height, max_height, seed_array)
        
        if instance[node.attribute] <= node.value:
            node.left = insert(node.left, instance, max_height, seed_array)
        else:
            node.right = insert(node.right, instance, max_height, seed_array)
    else:
        if node.height == max_height:
            node.data = np.vstack((node.data, instance))
        else:
            return RHT_build(np.vstack((node.data, instance)), node.height, max_height, seed_array)
    return node

# Score Calculation for a Single Instance (Equation 4)
def score_instance(tree, instance, total_instances):
    """
    Calculates the anomaly score of an instance in a single tree.

    Args:
        tree: The root of the tree (Node object).
        instance: The instance to score.
        total_instances: The total number of instances seen by the tree.

    Returns:
        The anomaly score of the instance based on Shannon's Information Content.
    """
    node = tree
    # Traverse the tree until reaching the appropriate leaf
    while not node.is_leaf():
        if instance[node.attribute] <= node.value:
            node = node.left
        else:
            node = node.right

    # Get the size of the leaf (distinct instances)
    leaf_size = len(np.unique(node.data, axis=0))  # Use unique instances in the leaf

    # Calculate P(Q) = |S(Q)| / n
    P_Q = leaf_size / total_instances

    # Calculate the anomaly score: log(1 / P(Q)) = log(n / |S(Q)|)
    anomaly_score = np.log(1 / (P_Q + 1e-10))  # Adding epsilon to prevent division by zero
    #print(f"Anomaly score for the instance: {anomaly_score}")
    return anomaly_score


# Random Histogram Forest (RHF)
class RandomHistogramForest:
    def __init__(self, num_trees, max_height, window_size, number_of_features):
        self.num_trees = num_trees
        self.max_height = max_height
        self.window_size = window_size
        self.forest = []
        self.seed_arrays = []
        self.reference_window = []  # Reference window for concept drift
        self.current_window = []  # Current window of incoming instances
        self.number_of_features = number_of_features
    
    def initialize_forest(self):
        self.forest = []
        #This creates an array of random integers between 0 and 10,000.
        #The length of each array is 2 ** self.max_height, which is the number of potential leafs in a complete binary tree of height max_height. 
        self.seed_arrays = [np.random.randint(0, 10000, 2 ** self.max_height) for _ in range(self.num_trees)]

        for i in range(self.num_trees):
            tree = RHT_build(np.empty((0, self.number_of_features)), 0, self.max_height, self.seed_arrays[i])
            self.forest.append(tree)
    
    def update_forest(self, instance):
        # Add the instance to the current window
        self.current_window.append(instance)
        
        # If the current window exceeds the window size, update the reference window
        if len(self.current_window) >= self.window_size:
            print(len(self.current_window))
            print('emptying current window')
            # Update the reference window with the current window data
            self.reference_window = self.current_window[-self.window_size:]
            self.current_window = []
            
            # Rebuild the forest directly from the reference window
            self.forest = []
            for i in range(self.num_trees):
                # Each tree is built directly using the reference window
                tree = RHT_build(np.array(self.reference_window), 0, self.max_height, self.seed_arrays[i])
                self.forest.append(tree)
        
        # Insert the new instance into the forest (each tree)
        for i, tree in enumerate(self.forest):
            self.forest[i] = insert(tree, instance, self.max_height, self.seed_arrays[i])
    
    #def score(self, instance):
    #    return np.sum([score_instance(tree, instance) for tree in self.forest])
    def score(self, instance):
        """
        Calculates the anomaly score for an instance across the entire forest.

        Args:
            instance: The instance to score.

        Returns:
            The aggregated anomaly score across all trees.
        """
        total_instances = sum(len(np.unique(tree.data, axis=0)) for tree in self.forest)  # Total instances across the forest
        return np.sum([score_instance(tree, instance, total_instances) for tree in self.forest])
    

# STREAMRHF (Algorithm 2)
def STREAMRHF(data_stream, max_height, num_trees, window_size, number_of_features):
    scores = []
    forest = RandomHistogramForest(num_trees, max_height, window_size, 
                                   number_of_features) 
    forest.initialize_forest()
    
    for idx, instance in enumerate(data_stream):        
        # Step 1: Update the forest with the new instance
        forest.update_forest(instance)

        # Step 2: Calculate the anomaly score for the current instance
        scores.append(forest.score(instance))

        # Print progress every 200 instances
        if (idx + 1) % 10 == 0:  # idx starts at 0, so add 1 for a human-readable count
            print(f"Processed {idx + 1} instances...")

    return scores

# Example of how to use the STREAMRHF

dataset_name = "annthyroid"
path = f"C:/Users/giova/Downloads/wetransfer_forstefan_2024-12-13_1400/forStefan/data/public/{dataset_name}.gz"
df = pd.read_csv(path)

# Generate synthetic data
#number_of_instances = 100
labels = df['label'].to_numpy(dtype='float32')#[:number_of_instances]
data_stream = df.drop('label', axis=1).to_numpy(dtype='float32')#[:number_of_instances]
print("data: ", data_stream.shape)

#np.random.seed(42)  # Set a seed for reproducibility
shuffled_indices = np.random.permutation(len(data_stream))
data_stream = data_stream[shuffled_indices]
labels = labels[shuffled_indices]

# Parameters
max_height = 5
num_trees = 100
window_size = len(data_stream) // 100  # 1% of the data stream

n_runs = 1  # number of runs
ap_scores = []
execution_times = []

for i in range(n_runs):
    # Measure Execution Time
    start_time = time.time()

    # Run STREAMRHF
    anomaly_scores = STREAMRHF(data_stream, max_height, num_trees, window_size, data_stream.shape[1])

    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

    # Calculate Average Precision Score
    ap_score = average_precision_score(labels, anomaly_scores)
    print(f"Run {i + 1}: AP = {ap_score:.4f}, Execution Time = {execution_time:.2f} seconds")
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
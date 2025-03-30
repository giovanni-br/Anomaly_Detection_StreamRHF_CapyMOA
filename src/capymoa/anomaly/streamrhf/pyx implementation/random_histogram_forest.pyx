import numpy as np
from scipy.stats import kurtosis
import random
import time
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
import pandas as pd
from scipy.stats import sem

# Random Histogram Tree Node
class Node:
    def __init__(self, data, height, max_height, seed, node_id):
        self.data = data  # Data at this node
        self.height = height  # Current depth of the tree
        self.max_height = max_height  # Maximum allowed height of the tree
        self.seed = seed  # Random seed for attribute selection
        self.attribute = None  # Split attribute
        self.value = None  # Split value
        self.left = None  # Left child
        self.right = None  # Right child
        self.node_id = node_id  # Unique node identifier

    def is_leaf(self):
        return self.left is None and self.right is None

def compute_kurtosis(data):
    data = np.asarray(data)
    kurtosis_values = np.zeros(data.shape[1])
    for feature_idx in range(data.shape[1]):
        feature_data = data[:, feature_idx]
        mean = np.mean(feature_data)
        variance = np.mean((feature_data - mean) ** 2)
        fourth_moment = np.mean((feature_data - mean) ** 4)
        kurtosis_value = fourth_moment / ((variance + 1e-10) ** 2)
        kurtosis_values[feature_idx] = np.log(kurtosis_value + 1)
    return kurtosis_values

def choose_split_attribute(kurt_values, random_seed):
    np.random.seed(int(random_seed))  # Ensure seed is an integer
    Ks = np.sum(kurt_values)
    r = np.random.uniform(0, Ks)
    cumulative = 0
    for idx, k_value in enumerate(kurt_values):
        cumulative += k_value
        if cumulative > r:
            return idx
    return len(kurt_values) - 1

def RHT_build(data, height, max_height, seed_array, node_id=1):
    node = Node(data, height, max_height, seed_array[node_id], node_id)
    if height == max_height or len(data) <= 1:
        return node

    kurt_values = compute_kurtosis(data)
    attribute = choose_split_attribute(kurt_values, node.seed)
    split_value = np.random.uniform(np.min(data[:, attribute]), np.max(data[:, attribute]))

    node.attribute = attribute
    node.value = split_value

    left_data = data[data[:, attribute] <= split_value]
    right_data = data[data[:, attribute] > split_value]

    node.left = RHT_build(left_data, height + 1, max_height, seed_array, node_id=2*node_id)
    node.right = RHT_build(right_data, height + 1, max_height, seed_array, node_id=(2*node_id)+1)
    return node

#I think we have to pay more attention to the node_id's here
def insert(node, instance, max_height, seed_array):
    if not node.is_leaf():
        kurt_values = compute_kurtosis(np.vstack((node.data, instance)))
        #new_attribute = choose_split_attribute(kurt_values, seed_array[node.height])
        new_attribute = choose_split_attribute(kurt_values, seed_array[node.node_id])  # Use the correct seed

        if node.attribute != new_attribute:
            # Attribute mismatch; rebuild subtree from this node
            #return RHT_build(np.vstack((node.data, instance)), node.height, max_height, seed_array, node_id=1)
            return RHT_build(np.vstack((node.data, instance)), node.height, max_height, seed_array, node_id=node.node_id)

        if instance[node.attribute] <= node.value:
            node.left = insert(node.left, instance, max_height, seed_array)
        else:
            node.right = insert(node.right, instance, max_height, seed_array)
    else:
        if node.height == max_height:
            node.data = np.vstack((node.data, instance))
        else:
            # Since the max height has not been reached, we can continue to build the tree
            #return RHT_build(np.vstack((node.data, instance)), node.height, max_height, seed_array, node_id=1)
            return RHT_build(np.vstack((node.data, instance)), node.height, max_height, seed_array, node_id=node.node_id)
    return node

def score_instance(tree, instance, total_instances):
    node = tree
    while not node.is_leaf():
        if instance[node.attribute] <= node.value:
            node = node.left
        else:
            node = node.right

    leaf_size = len(np.unique(node.data, axis=0))
    P_Q = leaf_size / total_instances
    anomaly_score = np.log(1 / (P_Q + 1e-10))
    return anomaly_score

class RandomHistogramForest:
    def __init__(self, num_trees, max_height, window_size, number_of_features):
        self.num_trees = num_trees
        self.max_height = max_height
        self.window_size = window_size
        self.forest = []
        self.seed_arrays = []
        self.reference_window = []
        self.current_window = []
        self.number_of_features = number_of_features

    def initialize_forest(self):
        self.forest = []
        # Maximum possible nodes in a full binary tree
        num_nodes = 2 ** (self.max_height + 1)
        # Each tree gets a seed array for all possible nodes
        self.seed_arrays = [np.random.randint(0, 10000, size=num_nodes) for _ in range(self.num_trees)]

        for i in range(self.num_trees):
            tree = RHT_build(np.empty((0, self.number_of_features)), 0, self.max_height, self.seed_arrays[i], node_id=1)
            self.forest.append(tree)

    def update_forest(self, instance):
        self.current_window.append(instance)

        if len(self.current_window) >= self.window_size:
            self.reference_window = self.current_window[-self.window_size:]
            self.current_window = []
            self.forest = []
            for i in range(self.num_trees):
                tree = RHT_build(np.array(self.reference_window), 0, self.max_height, self.seed_arrays[i], node_id=1)
                self.forest.append(tree)

        for i, tree in enumerate(self.forest):
            self.forest[i] = insert(tree, instance, self.max_height, self.seed_arrays[i])

    def score(self, instance):
        total_instances = sum(len(np.unique(tree.data, axis=0)) for tree in self.forest)
        return np.sum([score_instance(tree, instance, total_instances) for tree in self.forest])

def STREAMRHF(data_stream, max_height, num_trees, window_size, number_of_features):
    scores = []
    forest = RandomHistogramForest(num_trees, max_height, window_size, number_of_features)
    forest.initialize_forest()

    for idx, instance in enumerate(data_stream):
        forest.update_forest(instance)
        scores.append(forest.score(instance))
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} instances...")

    return scores
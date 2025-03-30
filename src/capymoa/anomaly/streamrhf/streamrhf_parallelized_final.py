import numpy as np
import pandas as pd
import time
from scipy.stats import kurtosis, sem
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
import multiprocessing

# ------------------------------------------------------------------------------------
# Random Histogram Tree Node
# ------------------------------------------------------------------------------------
class Node:
    def __init__(self, data, height, max_height, seed, node_id):
        self.data = data
        self.height = height
        self.max_height = max_height
        self.seed = seed
        self.attribute = None
        self.value = None
        self.left = None
        self.right = None
        self.node_id = node_id

    def is_leaf(self):
        return self.left is None and self.right is None

# ------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------
def collect_subtree_data(node, number_of_features):
    if node is None:
        return np.empty((0, number_of_features))
    collected_data = node.data if node.data is not None else np.empty((0, number_of_features))
    if not node.is_leaf():
        left_data = collect_subtree_data(node.left, number_of_features)
        right_data = collect_subtree_data(node.right, number_of_features)
        collected_data = np.vstack((collected_data, left_data, right_data))
    return collected_data

def compute_kurtosis(data):
    if len(data) == 0:
        return np.zeros(data.shape[1])
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
    np.random.seed(int(random_seed))
    Ks = np.sum(kurt_values)
    r = np.random.uniform(0, Ks)
    cumulative = 0
    for idx, k_value in enumerate(kurt_values):
        cumulative += k_value
        if cumulative > r:
            return idx
    return len(kurt_values) - 1

def RHT_build(data, height, max_height, seed_array, node_id=1):
    """
    Build or rebuild a subtree from data, up to max_height.
    """
    node = Node(None, height, max_height, seed_array[node_id], node_id)
    if height == max_height or len(data) <= 1:
        node.data = data
        return node

    kurt_values = compute_kurtosis(data)
    attribute = choose_split_attribute(kurt_values, node.seed)
    split_value = np.random.uniform(np.min(data[:, attribute]), np.max(data[:, attribute]))

    node.attribute = attribute
    node.value = split_value

    left_data = data[data[:, attribute] <= split_value]
    right_data = data[data[:, attribute] > split_value]

    node.left = RHT_build(left_data, height + 1, max_height, seed_array, node_id=2 * node_id)
    node.right = RHT_build(right_data, height + 1, max_height, seed_array, node_id=(2 * node_id) + 1)
    return node

def insert(node, instance, max_height, seed_array):
    """
    Insert one instance into the given node (subtree).
    Potentially rebuild subtree if an attribute mismatch occurs.
    """
    if not node.is_leaf():
        data_to_send = np.vstack((node.data, instance)) if node.data is not None else np.array([instance])
        kurt_values = compute_kurtosis(data_to_send)
        new_attribute = choose_split_attribute(kurt_values, seed_array[node.node_id])

        if node.attribute != new_attribute:
            subtree_data = collect_subtree_data(node, data_to_send.shape[1])
            return RHT_build(
                np.vstack((subtree_data, instance)),
                node.height,
                max_height,
                seed_array,
                node_id=node.node_id
            )

        if instance[node.attribute] <= node.value:
            node.left = insert(node.left, instance, max_height, seed_array)
        else:
            node.right = insert(node.right, instance, max_height, seed_array)
    else:
        data_to_send = np.vstack((node.data, instance)) if node.data is not None else np.array([instance])
        if node.height == max_height:
            node.data = data_to_send
        else:
            return RHT_build(data_to_send, node.height, max_height, seed_array, node_id=node.node_id)
    return node

def score_instance(tree, instance, total_instances):
    """
    Navigate down the tree to find the leaf, compute anomaly score.
    """
    node = tree
    while not node.is_leaf():
        if instance[node.attribute] <= node.value:
            node = node.left
        else:
            node = node.right

    leaf_size = len(node.data)
    if (total_instances == 0) or (leaf_size == 0):
        return 1
    if total_instances == leaf_size:
        return 0
    P_Q = leaf_size / total_instances
    return np.log(1 / (P_Q + 1e-10))

# ------------------------------------------------------------------------------------
# RandomHistogramForest with a Long-Lived Pool
# ------------------------------------------------------------------------------------
class RandomHistogramForest:
    def __init__(self, num_trees, max_height, window_size, number_of_features):
        print(window_size)
        self.num_trees = num_trees
        self.max_height = max_height
        self.window_size = window_size
        self.forest = []
        self.seed_arrays = []
        self.reference_window = []
        self.current_window = []
        self.number_of_features = number_of_features

        # Create a single, long-lived pool of worker processes
        self.pool = multiprocessing.Pool(processes=None)  # None => use all cores

    def close_pool(self):
        """Call this at the very end of the stream to cleanly exit worker processes."""
        self.pool.close()
        self.pool.join()

    # ----------------------
    # Build Forest
    # ----------------------
    def initialize_forest(self):
        self.forest = []
        num_nodes = 2 ** (self.max_height + 1)
        self.seed_arrays = [np.random.randint(0, 10000, size=num_nodes) for _ in range(self.num_trees)]

        # Prepare arguments
        build_args = [
            (np.empty((0, self.number_of_features)), 0, self.max_height, self.seed_arrays[i], 1)
            for i in range(self.num_trees)
        ]

        # Use starmap on the single pool
        self.forest = self.pool.starmap(RHT_build, build_args)

    # ----------------------
    # Rebuild Forest
    # ----------------------
    def rebuild_forest_on_reference(self):
        ref_data = np.array(self.reference_window)

        # Prepare arguments
        rebuild_args = [
            (ref_data, 0, self.max_height, self.seed_arrays[i], 1)
            for i in range(self.num_trees)
        ]

        # Rebuild in parallel
        self.forest = self.pool.starmap(RHT_build, rebuild_args)

    # ----------------------
    # Update with One Instance
    # ----------------------
    def update_forest(self, instance):
        self.current_window.append(instance)

        # If we reached the window size, rebuild from reference
        if len(self.current_window) >= self.window_size:
            self.reference_window = self.current_window[-self.window_size:]
            self.current_window = []
            self.rebuild_forest_on_reference()

        # Insert the instance into each tree
        # We'll starmap the 'insert_one_tree' helper below
        insert_args = [
            (self.forest[i], instance, self.max_height, self.seed_arrays[i])
            for i in range(self.num_trees)
        ]

        # Map in parallel
        updated_trees = self.pool.starmap(insert, insert_args)
        self.forest = updated_trees

    # ----------------------
    # Score One Instance
    # ----------------------
    def score(self, instance):
        total_instances = len(self.current_window) + len(self.reference_window)

        score_args = [(tree, instance, total_instances) for tree in self.forest]
        scores = self.pool.starmap(score_instance, score_args)
        return np.sum(scores)

# ------------------------------------------------------------------------------------
# Streaming Function
# ------------------------------------------------------------------------------------
def STREAMRHF(data_stream, max_height, num_trees, window_size, number_of_features):
    scores = []
    forest = RandomHistogramForest(num_trees, max_height, window_size, number_of_features)
    forest.initialize_forest()

    for idx, instance in enumerate(data_stream):
        forest.update_forest(instance)
        scores.append(forest.score(instance))
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} instances...")

    # Be sure to close the pool once we're done
    forest.close_pool()
    return scores

if __name__ == "__main__":
    print("Starting...")

    np.random.seed(42)

    dataset_names = ["smtp_all"]

    all_results = []

    for dataset_name in dataset_names:
    
        print(f"Dataset: {dataset_name}")

        path = f"/home/infres/benedetti-23/CapyMOA/public/{dataset_name}.gz"
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

        n_runs = 2
        ap_scores = []
        execution_times = []

        dataset_results = []

        for i in range(n_runs):
            start_time = time.time()

            # Run STREAMRHF with one persistent pool
            anomaly_scores = STREAMRHF(data_stream, max_height, num_trees, window_size, data_stream.shape[1])

            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)

            # Evaluate
            ap_score = average_precision_score(labels, anomaly_scores)
            auc_score = roc_auc_score(labels, anomaly_scores)
            fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
            auc_paper = auc(fpr, tpr)

            print(f"Run {i + 1}: AP = {ap_score:.4f}, Execution Time = {execution_time:.2f} seconds")
            print(f"AUC Score from sklearn: {auc_score:.4f}")
            print(f"AUC Score from paper: {auc_paper:.4f}")
            
            # Save individual run results
            run_result = {
                'Dataset': dataset_name,
                'Run': i + 1,
                'AP': ap_score,
                'Execution Time (s)': execution_time,
                'AUC (sklearn)': auc_score,
                'AUC (paper)': auc_paper
            }
            dataset_results.append(run_result)
            all_results.append(run_result)

            # Save checkpoint after each run
            results_df = pd.DataFrame(all_results)
            results_df.to_csv("all_run_results_checkpoint.csv", index=False)
            print(f"Checkpoint saved after Run {i + 1}")

        # Summaries
        ap_scores = np.array(ap_scores)
        execution_times = np.array(execution_times)

        mean_ap = np.mean(ap_scores)
        mean_time = np.mean(execution_times)
        ap_sem = sem(ap_scores)
        time_sem = sem(execution_times)
        confidence_level = 1.96
        ap_ci = confidence_level * ap_sem
        time_ci = confidence_level * time_sem

        print(f"Over {n_runs} runs:")
        print(f"AP: {mean_ap:.4f} ± {ap_ci:.4f} (95% CI)")
        print(f"Execution Time: {mean_time:.2f} ± {time_ci:.2f} seconds (95% CI)")

        # Save summary
        dataset_summary = {
            'Dataset': dataset_name,
            'Metric': ['AP', 'Execution Time'],
            'Mean': [mean_ap, mean_time],
            'CI (95%)': [ap_ci, time_ci]
        }

        summary_df = pd.DataFrame(dataset_summary)
        summary_df.to_csv(f"{dataset_name}_summary.csv", index=False)

    # Save all results to a final CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("all_run_results.csv", index=False)
    print("Final results saved to 'all_run_results.csv'")
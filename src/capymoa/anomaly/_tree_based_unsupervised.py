import numpy as np
import random
import math
from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.type_alias import AnomalyScore

class TreeBasedUnsupervised(AnomalyDetector):
    def __init__(self, schema=None, num_trees=40, max_height=20, window_size=100, random_seed=1):
        super().__init__(schema, random_seed=random_seed)
        self.num_trees = num_trees
        self.max_height = max_height
        self.window_size = window_size
        self.random_seed = random_seed

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self.forest = []
        self.reference_window = []
        self.current_window = []

        self._initialize_forest()
    def Insert(self, tree, i, h, z, node_index=0):
        if node_index in tree["leaves"]:
            # Leaf node
            tree["leaves"][node_index] += 1
            return tree
        
        split_info = tree["splits"].get(node_index)
        if not split_info:
            # New node
            new_tree = {
                "splits": {},
                "leaves": {node_index: 1},
                "r_value": random.uniform(0, 1)
            }
            return self._build_tree(new_tree, np.array([i]), 0, node_index)

        attribute = split_info["attribute"]
        value = split_info["value"]

        if i[attribute] <= value:
            tree["splits"][node_index] = split_info
            tree["left"] = self.Insert(tree, i, h, z, 2 * node_index + 1)
        else:
            tree["splits"][node_index] = split_info
            tree["right"] = self.Insert(tree, i, h, z, 2 * node_index + 2)

        return tree

    def _initialize_forest(self):
        """Initialize a forest of Random Histogram Trees (RHTs)."""
        for _ in range(self.num_trees):
            tree = {
                "splits": {},  # Node split information
                "leaves": {},  # Leaf statistics
                "r_value": random.uniform(0, 1),  # Random value for split attribute
            }
            self.forest.append(tree)

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis for a given feature."""
        if len(data) <= 1:
            return 0

        mean = np.mean(data)
        variance = np.var(data)
        if variance == 0:
            return 0

        fourth_moment = np.mean((data - mean) ** 4)
        kurtosis = fourth_moment / (variance ** 2)
        return kurtosis

    def _kurtosis_split(self, data):
        """Select split attribute and split value based on kurtosis."""
        num_features = data.shape[1]
        kurtosis_scores = [self._calculate_kurtosis(data[:, i]) for i in range(num_features)]

        # Compute cumulative sum of log-transformed kurtosis scores
        cumulative_kurtosis = np.cumsum([np.log(k + 1) for k in kurtosis_scores])
        total_kurtosis = cumulative_kurtosis[-1]

        # Randomly select a split attribute
        r = random.uniform(0, total_kurtosis)
        split_attribute = np.searchsorted(cumulative_kurtosis, r)

        # Select a split value uniformly within the range of the selected attribute
        split_value = random.uniform(data[:, split_attribute].min(), data[:, split_attribute].max())

        return split_attribute, split_value

    def _build_tree(self, tree, data, depth=0, node_index=0):
        """Recursively build a Random Histogram Tree."""
        if depth == self.max_height or len(data) <= 1:
            # Leaf node
            tree["leaves"][node_index] = len(data)
            return tree

        # Select split attribute and split value
        split_attribute, split_value = self._kurtosis_split(data)

        # Split the data
        left_data = data[data[:, split_attribute] <= split_value]
        right_data = data[data[:, split_attribute] > split_value]

        # Store split information
        tree["splits"][node_index] = {
            "attribute": split_attribute,
            "value": split_value,
        }

        # Recursively build left and right subtrees
        tree = self.Insert(tree, left_data[0], depth + 1, 2 * node_index + 1)
        tree = self.Insert(tree, right_data[0], depth + 1, 2 * node_index + 2)

        return tree

    def _get_leaf_size(self, tree, instance, node_index=0):
        """Traverse the tree to find the leaf size for a given instance."""
        if node_index in tree["leaves"]:
            return tree["leaves"][node_index]

        split_info = tree["splits"].get(node_index)
        if not split_info:
            return 0

        attribute = split_info["attribute"]
        value = split_info["value"]

        if instance[attribute] <= value:
            return self._get_leaf_size(tree, instance, 2 * node_index + 1)
        else:
            return self._get_leaf_size(tree, instance, 2 * node_index + 2)

    def train(self, instance: Instance):
        """Train the model with a new instance."""
        instance_array = np.array(instance.x).reshape(1, -1)
        self.current_window.append(instance_array[0])

        if len(self.current_window) >= self.window_size:
            self.reference_window = np.array(self.current_window)
            self.current_window = []

            # Rebuild the forest
            self.forest = []
            self._initialize_forest()
            for tree in self.forest:
                self._build_tree(tree, self.reference_window)

    def score_instance(self, instance: Instance) -> AnomalyScore:
        instance_array = np.array(instance.x)
        scores = []

        for tree in self.forest:
            leaf_size = self._get_leaf_size(tree, instance_array)
            if leaf_size > 0:
                scores.append(math.log(1 / leaf_size))

        anomaly_score = sum(scores) if scores else 0
        return anomaly_score


    def predict(self, instance: Instance) -> AnomalyScore:
        """Return the raw anomaly score."""
        return self.score_instance(instance)

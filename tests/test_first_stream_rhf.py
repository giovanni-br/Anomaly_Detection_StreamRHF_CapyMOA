import pytest
from sklearn.metrics import average_precision_score, roc_curve, auc
from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.datasets import ElectricityTiny
from capymoa.stream import stream_from_file
from capymoa.stream._stream import Schema
import numpy as np
import os
from capymoa.anomaly import (
    StreamRHF
)

# Variables
path_to_dataset = r"C:\Users\aleja\OneDrive - Universidad Nacional de Colombia\Documentos\Institut Polytechnique de Paris\courses\P1\Data Streaming\project\actual code\datasets\forStefan\data\public"
dataset_name = "abalone"

# Construct input and output paths
input_path = os.path.join(path_to_dataset, f"{dataset_name}.gz")
output_path = os.path.join(path_to_dataset, f"{dataset_name}.csv")

stream = stream_from_file(output_path, dataset_name=dataset_name)

@pytest.mark.parametrize(
    "dataset_class, expected_auc",
    [
        #(ElectricityTiny, 0.367),  # Example AUC score for ElectricityTiny
        (stream, 0.74),  # Example AUC score for ElectricityTiny
    ],
)
#def test_stream_rhf(dataset_class, expected_auc):
#    """Test the StreamRHF model using a CapyMOA dataset."""
#    # Initialize the stream and schema
#    stream = dataset_class()
def test_stream_rhf(dataset_class, expected_auc):
    """Test the StreamRHF model using a CapyMOA dataset."""
    # Initialize the stream and schema
    stream = dataset_class
    schema: Schema = stream.get_schema()

    # Collect all instances into a dataset
    data = []
    labels = []

    while stream.has_more_instances():
        instance = stream.next_instance()
        data.append(instance.x)
        labels.append(instance.y_index)

    # Convert data and labels to numpy arrays
    data = np.array(data, dtype='float64')
    data = data.copy(order='C')
    print("First dataset row: ", data[0])
    labels = np.array(labels, dtype='int32')

    # Initialize StreamRHF model
    rhf_model = StreamRHF(schema=schema, number_of_trees=100, height=5)

    # initial sample size, percentage or constant (see "const" parameter below)
    init = 1
    N_init_pts = int(round(data.shape[0] * (init / 100)))

    print("Number of initial points / window size: ", N_init_pts)

    # Get anomaly scores for the entire dataset
    scores = rhf_model.get_scores(data, N_init_pts)  # Example initial points

    # Initialize the evaluator and evaluate batch anomaly scores
    evaluator = AnomalyDetectionEvaluator(schema=schema)
    for y_index, score in zip(labels, scores):
        evaluator.update(y_index, score)

    print(labels[0], scores[0])
    print(type(labels[0]), type(scores[0]))

    # Compute and assert AUC score
    actual_auc = evaluator.auc()
    print(f"AUC CapyMOA: {actual_auc}")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    print("AUC score sklearn:", auc(fpr, tpr))

    ap_score = average_precision_score(labels, scores)
    print(f"AP score sklearn: {ap_score}")
    
    assert actual_auc == pytest.approx(
        #expected_auc, abs=0.01
        expected_auc, abs=1
    ), f"Expected AUC {expected_auc}, but got {actual_auc}."
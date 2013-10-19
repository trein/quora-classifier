import numpy as np
import re

def extract_all(filename):
    return extract(filename)

def extract_selected_all(filename):
    matrix_features, vector_targets = extract_all(filename)
    sel_matrix_features = np.delete(matrix_features, [0, 8, 9, 2, 21, 22], 1)

    return (sel_matrix_features, vector_targets)

def extract_normalized_all(filename):
    matrix_features, vector_targets = extract_all(filename)
    normal_matrix_features = normalize_features(matrix_features)

    return (normal_matrix_features, vector_targets)

def extract_normalized_selected_all(filename):
    matrix_features, vector_targets = extract_normalized_all(filename)
    sel_matrix_features = np.delete(matrix_features, [0, 8, 9, 2, 21, 22], 1)

    return (sel_matrix_features, vector_targets)

def extract(file):
    input_file = open(file)
    traindata = input_file.readlines()
    features = []
    targets = []

    for line in traindata:
        formatted_line = line.strip("\n")
        target_i = formatted_line.split(" ")[1]
        feature_i = re.sub(r"(\d+):", "", formatted_line).split(" ")[2:]

        targets.append(target_i)
        features.append(feature_i)

    matrix_features = np.array(features).astype(np.float)
    vector_targets = np.array(targets).astype(np.int)

    return (matrix_features, vector_targets)

def normalize_features(matrix_features):
    max_features = matrix_features.max(axis = 0)
    max_features = (max_features + (max_features == 0))

    return matrix_features / max_features

def accuracy(targets_hat, targets):
    return (1.0 * (targets_hat == targets)).sum(0) / targets.shape
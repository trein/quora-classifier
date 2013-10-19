import numpy as np
import re

def extract_train():
    return extract('dataset/data.txt', [0, 3500])

def extract_test():
    return extract('dataset/data.txt', [3501, 4500])

def extract_all():
    return extract('dataset/data.txt', [0, 4500])

def extract_normalized_train():
    matrix_features, vector_targets = extract_train()
    normal_matrix_features = normalize_features(matrix_features)

    return (normal_matrix_features, vector_targets)

def extract_normalized_test():
    matrix_features, vector_targets = extract_test()
    normal_matrix_features = normalize_features(matrix_features)

    return (normal_matrix_features, vector_targets)

def extract_selected_train():
    matrix_features, vector_targets = extract_normalized_train()
    sel_matrix_features = np.delete(matrix_features, [21, 22], 1)

    return (sel_matrix_features, vector_targets)

def extract_selected_test():
    matrix_features, vector_targets = extract_normalized_test()
    sel_matrix_features = np.delete(matrix_features, [21, 22], 1)
    
    return (sel_matrix_features, vector_targets)

def extract(file, index):
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

    return (matrix_features[index[0]:index[1]], vector_targets[index[0]:index[1]])

def normalize_features(matrix_features):
    max_features = matrix_features.max(axis = 0)
    max_features = (max_features + (max_features == 0))

    return matrix_features / max_features

def accuracy(targets_hat, targets):
    return (1.0 * (targets_hat == targets)).sum(0) / targets.shape
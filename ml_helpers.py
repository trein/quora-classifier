import numpy as np
import re

def extract_train():
    return extract('dataset/train.txt')

def extract_test():
    return extract('dataset/test.txt')

def extract(file):
    input_file = open(file)
    traindata = input_file.readlines()
    features = []
    targets = []

    for line in traindata:
        formatted_line = line.replace("\n", "")
        target_i = formatted_line.split(" ")[1]
        feature_i = re.sub(r"(\d+):", "", formatted_line).split(" ")[2:]

        targets.append(target_i)
        features.append(feature_i)

    matrix_features = np.array(features).astype(np.float)

    max_features = matrix_features.max(axis = 0)
    max_features = (max_features + (max_features == 0))

    normal_matrix_features = matrix_features / max_features
    vector_targets = np.array(targets).astype(np.int)

    # print normal_matrix_features
    # print "Max", max_features

    return (normal_matrix_features, vector_targets)

def accuracy(targets_hat, targets):
    return (1.0 * (targets_hat == targets)).sum(0) / targets.shape
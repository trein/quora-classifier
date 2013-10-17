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

    return (np.array(features).astype(np.float), np.array(targets).astype(np.int))
from sklearn import linear_model
import numpy as np

EXCLUDED_FEATURES = [0, 8, 9, 2, 21, 22]

def extract_lasso_features_indexes(matrix_features, vector_targets):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(matrix_features, vector_targets)

    return [i for i, e in enumerate(clf.coef_) if e == 0]

def extract_features(included_index, matrix_features, vector_targets):
    return (matrix_features[:,included_index], vector_targets)

def extract_empirical_features(matrix_features, vector_targets):
    sel_matrix_features = np.delete(matrix_features, EXCLUDED_FEATURES, 1)
    return (sel_matrix_features, vector_targets)

def extract_norm(matrix_features, vector_targets):
    normal_matrix_features = normalize_features(matrix_features)
    return (normal_matrix_features, vector_targets)

def extract_empirical_features_norm(matrix_features, vector_targets):
    sel_matrix_features = np.delete(matrix_features, EXCLUDED_FEATURES, 1)
    return (sel_matrix_features, vector_targets)

def normalize_features(matrix_features):
    max_features = matrix_features.max(axis = 0)
    max_features = (max_features + (max_features == 0))

    return matrix_features / max_features


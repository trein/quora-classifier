from sklearn import linear_model
from sklearn.svm import LinearSVC
import numpy as np


EXCLUDED_FEATURES = [0, 8, 9, 2, 21, 22]
RF_INCLUDED_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]
FOBA_INCLUDED_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]


def extract_lasso_features_indexes(matrix_features, vector_targets):
    """
    Perform Lasso feature selection.
    """

    clf = linear_model.Lasso(alpha=0.022, fit_intercept=False,
                             max_iter=2000,normalize=False, positive=False,
                             tol=0.001, warm_start=True)
    clf.fit(matrix_features, vector_targets)

    return [i for i, e in enumerate(clf.coef_) if e != 0 and abs(e) > 1e-6]


def extract_linear_features_indexes(matrix_features, vector_targets):
    """
    Perform feature selection using a simple linear classifier.
    """

    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
    clf.fit(matrix_features, vector_targets)

    return [i for i, e in enumerate(clf.coef_[0]) if e != 0 and abs(e) > 1e-6]


def extract_foba_features_indexes():
    """
    Return features to be included according FOBA algorithm (algorithm
    executed offline and results added here).
    """

    return FOBA_INCLUDED_FEATURES


def extract_rf_features_indexes():
    """
    Return features to be included according RF algorithm (algorithm
    executed offline and results added here).
    """

    return RF_INCLUDED_FEATURES


def extract_features(included_index, matrix_features, vector_targets):
    """
    Return the only features that must be included in the classification
    process.
    """

    return matrix_features[:, included_index], vector_targets


def extract_empirical_features(matrix_features, vector_targets):
    """
    Return relevant features after performing a simple Forward-Backward
    not-Greedy feature selection.
    """

    sel_matrix_features = np.delete(matrix_features, EXCLUDED_FEATURES, 1)
    return sel_matrix_features, vector_targets


def extract_norm(matrix_features, vector_targets):
    """
    Return normalized dataset.
    """

    normal_matrix_features = normalize_features(matrix_features)
    return normal_matrix_features, vector_targets


def extract_empirical_features_norm(matrix_features, vector_targets):
    """
    Return normalized dataset after performing a simple Forward-Backward
    not-Greedy feature selection.
    """

    normal_matrix_features = normalize_features(matrix_features)
    sel_matrix_features = np.delete(normal_matrix_features, EXCLUDED_FEATURES, 1)
    return sel_matrix_features, vector_targets


def normalize_features(matrix_features):
    """
    Perform dataset normalization.
    """

    max_features = matrix_features.max(axis = 0)
    max_features = (max_features + (max_features == 0))

    return matrix_features / max_features
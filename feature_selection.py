from sklearn import linear_model
from sklearn.svm import LinearSVC
import numpy as np

EXCLUDED_FEATURES = [0, 8, 9, 2, 21, 22]
RF_EXCLUDED_FEATURES = [19, 22, 23]
FOBA_EXCLUDED_FEATURES = [8, 19, 21, 22]

def extract_lasso_features_indexes(matrix_features, vector_targets):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(matrix_features, vector_targets)

    print clf.coef_

    return [i for i, e in enumerate(clf.coef_) if e != 0]

def extract_linear_features_indexes(matrix_features, vector_targets):
    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
    clf.fit(matrix_features, vector_targets)

    # print sel_matrix_features.shape
    print clf.coef_

    return [i for i, e in enumerate(clf.coef_[0]) if e != 0]

def extract_random_forest_features_indexes(matrix_features, vector_targets):
    normal_matrix_features = normalize_features(matrix_features)
    sel_matrix_features = np.delete(normal_matrix_features, RF_EXCLUDED_FEATURES, 1)
    return (sel_matrix_features, vector_targets)

def extract_foba_features_indexes(matrix_features, vector_targets):
    normal_matrix_features = normalize_features(matrix_features)
    sel_matrix_features = np.delete(normal_matrix_features, FOBA_EXCLUDED_FEATURES, 1)
    return (sel_matrix_features, vector_targets)

def extract_features(included_index, matrix_features, vector_targets):
    return (matrix_features[:,included_index], vector_targets)

def extract_empirical_features(matrix_features, vector_targets):
    sel_matrix_features = np.delete(matrix_features, EXCLUDED_FEATURES, 1)
    return (sel_matrix_features, vector_targets)

def extract_norm(matrix_features, vector_targets):
    normal_matrix_features = normalize_features(matrix_features)
    return (normal_matrix_features, vector_targets)

def extract_empirical_features_norm(matrix_features, vector_targets):
    normal_matrix_features = normalize_features(matrix_features)
    sel_matrix_features = np.delete(normal_matrix_features, EXCLUDED_FEATURES, 1)
    return (sel_matrix_features, vector_targets)

def normalize_features(matrix_features):
    max_features = matrix_features.max(axis = 0)
    # max_features = np.apply_along_axis(np.linalg.norm, 0, matrix_features)
    max_features = (max_features + (max_features == 0))

    return matrix_features / max_features
    # return whiten(matrix_features)

def whiten(X, fudge=1E-18):
   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d,V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1./np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V,D),V.T)

   # multiply by the whitening matrix
   X = np.dot(X,W)

   return X

import ml_helpers as mh
import feature_selection as selection
import quora_classifiers as qc
import quora_nnet as nnet
import quora_lr as lr
from time import time


# -------------------------------------------------------------------------
# TEST: Raw dataset (no modification)
# -------------------------------------------------------------------------
def test_raw_dataset(train_filename, test_filename):
    """
    Experiment considering raw dataset (no modification).
    """
    test_name = "raw dataset"
    (all_features, all_targets) = mh.extract(train_filename)
    (test_features, test_targets) = mh.extract(test_filename)
    test_classifiers(all_features, all_targets, test_features, test_targets, test_name)


# -------------------------------------------------------------------------
# TEST: Raw dataset and feature selection
# -------------------------------------------------------------------------
def test_selected_dataset(train_filename, test_filename):
    """
    Experiment considering dataset and feature selection.
    """
    test_name = "raw dataset with feature selection"
    (all_features, all_targets) = mh.extract(train_filename)
    (sel_features, sel_targets) = selection.extract_empirical_features(all_features, all_targets)

    (test_features, test_targets) = mh.extract(test_filename)
    (sel_test_features, sel_test_targets) = selection.extract_empirical_features(test_features, test_targets)

    test_classifiers(sel_features, sel_targets, sel_test_features, sel_test_targets, test_name)


# -------------------------------------------------------------------------
# TEST: Normalized dataset
# -------------------------------------------------------------------------
def test_normalized_dataset(train_filename, test_filename):
    """
    Experiment considering normalized dataset.
    """
    test_name = "normalized dataset"
    (all_features, all_targets) = mh.extract(train_filename)
    (sel_features, sel_targets) = selection.extract_norm(all_features, all_targets)

    (test_features, test_targets) = mh.extract(test_filename)
    (sel_test_features, sel_test_targets) = selection.extract_norm(test_features, test_targets)

    test_classifiers(sel_features, sel_targets, sel_test_features, sel_test_targets, test_name)


# -------------------------------------------------------------------------
# TEST: Normalized dataset and experimental feature selection
# -------------------------------------------------------------------------
def test_normalized_selected_dataset(train_filename, test_filename):
    """
    Experiment considering normalized dataset and experimental feature selection.
    """
    test_name = "normalized dataset and experimental feature selection"
    (all_features, all_targets) = mh.extract(train_filename)
    (sel_features, sel_targets) = selection.extract_empirical_features_norm(all_features, all_targets)

    (test_features, test_targets) = mh.extract(test_filename)
    (sel_test_features, sel_test_targets) = selection.extract_empirical_features_norm(test_features, test_targets)

    test_classifiers(sel_features, sel_targets, sel_test_features, sel_test_targets, test_name)


# -------------------------------------------------------------------------
# TEST: Normalized dataset and Random Forest feature selection
# -------------------------------------------------------------------------
def test_random_forest_selected_dataset(train_filename, test_filename):
    """
    Experiment considering normalized dataset and Random Forest feature selection.
    """
    test_name = "normalized and random forest feature selection"
    (all_features, all_targets) = mh.extract(train_filename)

    features_to_keep = selection.extract_rf_features_indexes()
    (norm_features, norm_targets) = selection.extract_norm(all_features, all_targets)
    (sel_features, sel_targets) = selection.extract_features(features_to_keep, norm_features, norm_targets)

    (test_features, test_targets) = mh.extract(test_filename)
    (norm_test_features, norm_test_targets) = selection.extract_norm(test_features, test_targets)
    (sel_test_features, sel_test_targets) = selection.extract_features(features_to_keep, norm_test_features, norm_test_targets)

    test_classifiers(sel_features, sel_targets, sel_test_features, sel_test_targets, test_name)
    print "RF to features keep:", features_to_keep


# -------------------------------------------------------------------------
# TEST: Normalized dataset and FOBA feature selection
# -------------------------------------------------------------------------
def test_foba_selected_dataset(train_filename, test_filename):
    """
    Experiment considering normalized dataset and FOBA feature selection.
    """
    test_name = "normalized and foba feature selection"
    (all_features, all_targets) = mh.extract(train_filename)

    features_to_keep = selection.extract_foba_features_indexes()
    (norm_features, norm_targets) = selection.extract_norm(all_features, all_targets)
    (sel_features, sel_targets) = selection.extract_features(features_to_keep, norm_features, norm_targets)

    (test_features, test_targets) = mh.extract(test_filename)
    (norm_valid_features, norm_valid_targets) = selection.extract_norm(test_features, test_targets)
    (sel_test_features, sel_test_targets) = selection.extract_features(features_to_keep, norm_valid_features, norm_valid_targets)

    test_classifiers(sel_features, sel_targets, sel_test_features, sel_test_targets, test_name)
    print "FOBA to features keep:", features_to_keep


# -------------------------------------------------------------------------
# TEST: Raw dataset and Lasso feature selection
# -------------------------------------------------------------------------
def test_lasso_selected_dataset(train_filename, test_filename):
    """
    Experiment considering raw dataset and Lasso feature selection.
    """
    test_name = "raw dataset with Lasso feature selection"
    (all_features, all_targets) = mh.extract(train_filename)

    features_to_keep = selection.extract_lasso_features_indexes(all_features, all_targets)
    (sel_features, sel_targets) = selection.extract_features(features_to_keep, all_features, all_targets)

    (test_features, test_targets) = mh.extract(test_filename)
    (sel_test_features, sel_test_targets) = selection.extract_features(features_to_keep, test_features, test_targets)

    test_classifiers(sel_features, sel_targets, sel_test_features, sel_test_targets, test_name)
    print "Lasso to features keep:", features_to_keep


# -------------------------------------------------------------------------
# TEST: Raw dataset and Linear feature selection
# -------------------------------------------------------------------------
def test_linear_selected_dataset(train_filename, test_filename):
    """
    Experiment considering raw dataset and Linear feature selection.
    """
    test_name = "normalized and Linear feature selection"
    (all_features, all_targets) = mh.extract(train_filename)

    features_to_keep = selection.extract_linear_features_indexes(all_features, all_targets)
    (sel_features, sel_targets) = selection.extract_features(features_to_keep, all_features, all_targets)

    (test_features, test_targets) = mh.extract(test_filename)
    (sel_test_features, sel_test_targets) = selection.extract_features(features_to_keep, test_features, test_targets)

    test_classifiers(sel_features, sel_targets, sel_test_features, sel_test_targets, test_name)
    print "Linear features to keep:", features_to_keep


def test_classifiers(features, targets, test_features, test_targets, test_name):
    """
    Evaluate classification accuracy considering the features/targets for training and validation.
    """
    classifiers = {
        "NB M" : qc.QuoraMultiNB(features, targets),
        "NB G": qc.QuoraGaussianNB(features, targets),
        "LR" : qc.QuoraLR(features, targets),
        "DT" : qc.QuoraDT(features, targets),
        "KNN" : qc.QuoraKNN(features, targets),
        "SVM" : qc.QuoraSVC(features, targets),
        "LDA" : qc.QuoraLDA(features, targets),
        "QDA" : qc.QuoraQDA(features, targets),
        "RFrst" : qc.QuoraRandomForest(features, targets),
        "ABoost" : qc.QuoraAdaBoost(features, targets),
        "Nnet" : nnet.QuoraNnet(features, targets),
        "ML-LR" : lr.QuoraMlLR(features, targets),
    }

    make_section("Test: %s" % test_name)

    for name, clf in classifiers.iteritems():
        start = time()

        clf.train()
        accuracy = clf.accuracy(test_features, test_targets)
        elapsed = time() - start

        print "%s \t %s \t (%.4f seconds)" % (name, accuracy, elapsed)
    print ""


def make_section(name):
    """
    Print section separator on console.
    """
    print "-"*80, "\n", name, "\n", "-"*80


def print_dataset_info(train_filename):
    """
    Print information regarding the provided dataset.
    """
    make_section("Dataset Information")

    (all_features, all_targets) = mh.extract(train_filename)

    print "Min values in features:", all_features.min(axis=0)
    print "Max values in features:", all_features.max(axis=0)


if __name__ == "__main__":
    train_dataset_filename = 'dataset/train.txt'
    test_filename = 'dataset/test.txt'

    print_dataset_info(train_dataset_filename)

    test_raw_dataset(train_dataset_filename, test_filename)
    test_selected_dataset(train_dataset_filename, test_filename)
    test_normalized_dataset(train_dataset_filename, test_filename)
    test_normalized_selected_dataset(train_dataset_filename, test_filename)
    test_foba_selected_dataset(train_dataset_filename, test_filename)
    test_random_forest_selected_dataset(train_dataset_filename, test_filename)
    test_lasso_selected_dataset(train_dataset_filename, test_filename)
    test_linear_selected_dataset(train_dataset_filename, test_filename)
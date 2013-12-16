import ml_helpers as mh
import feature_selection as selection
import quora_classifiers as qc
import quora_nnet as nnet
import quora_lr as lr
import quora_lr_t as lrt
from time import time

# -------------------------------------------------------------------------
# TEST: Raw dataset (no modification)
# -------------------------------------------------------------------------
def test_raw_dataset(train_filename, valid_filename):
    test_name = "raw dataset"
    (all_features, all_targets) = mh.extract(train_filename)
    (valid_features, valid_targets) = mh.extract(valid_filename)
    test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Raw dataset and feature selection
# -------------------------------------------------------------------------
def test_selected_dataset(train_filename, valid_filename):
    test_name = "raw dataset with feature selection"
    (all_features, all_targets) = mh.extract(train_filename)
    (sel_features, sel_targets) = selection.extract_empirical_features(all_features, all_targets)

    (valid_features, valid_targets) = mh.extract(valid_filename)
    (sel_valid_features, sel_valid_targets) = selection.extract_empirical_features(valid_features, valid_targets)

    test_classifiers(sel_features, sel_targets, sel_valid_features, sel_valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Normalized dataset
# -------------------------------------------------------------------------
def test_normalized_dataset(train_filename, valid_filename):
    test_name = "normalized dataset"
    (all_features, all_targets) = mh.extract(train_filename)
    (sel_features, sel_targets) = selection.extract_norm(all_features, all_targets)

    (valid_features, valid_targets) = mh.extract(valid_filename)
    (sel_valid_features, sel_valid_targets) = selection.extract_norm(valid_features, valid_targets)

    test_classifiers(sel_features, sel_targets, sel_valid_features, sel_valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Normalized dataset and experimental feature selection
# -------------------------------------------------------------------------
def test_normalized_selected_dataset(train_filename, valid_filename):
    test_name = "normalized dataset and experimental feature selection"
    (all_features, all_targets) = mh.extract(train_filename)
    (sel_features, sel_targets) = selection.extract_empirical_features_norm(all_features, all_targets)

    (valid_features, valid_targets) = mh.extract(valid_filename)
    (sel_valid_features, sel_valid_targets) = selection.extract_empirical_features_norm(valid_features, valid_targets)

    test_classifiers(sel_features, sel_targets, sel_valid_features, sel_valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Normalized dataset and Random Forest feature selection
# -------------------------------------------------------------------------
def test_random_forest_selected_dataset(train_filename, valid_filename):
    test_name = "normalized and random forest feature selection"
    (all_features, all_targets) = mh.extract(train_filename)

    features_to_keep = selection.extract_rf_features_indexes(all_features, all_targets)
    (norm_features, norm_targets) = selection.extract_norm(all_features, all_targets)
    (sel_features, sel_targets) = selection.extract_features(features_to_keep, norm_features, norm_targets)

    (valid_features, valid_targets) = mh.extract(valid_filename)
    (norm_valid_features, norm_valid_targets) = selection.extract_norm(valid_features, valid_targets)
    (sel_valid_features, sel_valid_targets) = selection.extract_features(features_to_keep, norm_valid_features, norm_valid_targets)

    test_classifiers(sel_features, sel_targets, sel_valid_features, sel_valid_targets, test_name)
    print "RF to features keep:", features_to_keep

# -------------------------------------------------------------------------
# TEST: Normalized dataset and FOBA feature selection
# -------------------------------------------------------------------------
def test_foba_selected_dataset(train_filename, valid_filename):
    test_name = "normalized and foba feature selection"
    (all_features, all_targets) = mh.extract(train_filename)

    features_to_keep = selection.extract_foba_features_indexes(all_features, all_targets)
    (norm_features, norm_targets) = selection.extract_norm(all_features, all_targets)
    (sel_features, sel_targets) = selection.extract_features(features_to_keep, norm_features, norm_targets)

    (valid_features, valid_targets) = mh.extract(valid_filename)
    (norm_valid_features, norm_valid_targets) = selection.extract_norm(valid_features, valid_targets)
    (sel_valid_features, sel_valid_targets) = selection.extract_features(features_to_keep, norm_valid_features, norm_valid_targets)

    test_classifiers(sel_features, sel_targets, sel_valid_features, sel_valid_targets, test_name)
    print "FOBA to features keep:", features_to_keep

# -------------------------------------------------------------------------
# TEST: Raw dataset and Lasso feature selection
# -------------------------------------------------------------------------
def test_lasso_selected_dataset(train_filename, valid_filename):
    test_name = "raw dataset with Lasso feature selection"
    (all_features, all_targets) = mh.extract(train_filename)

    features_to_keep = selection.extract_lasso_features_indexes(all_features, all_targets)
    (sel_features, sel_targets) = selection.extract_features(features_to_keep, all_features, all_targets)

    (valid_features, valid_targets) = mh.extract(valid_filename)
    (sel_valid_features, sel_valid_targets) = selection.extract_features(features_to_keep, valid_features, valid_targets)

    test_classifiers(sel_features, sel_targets, sel_valid_features, sel_valid_targets, test_name)
    print "Lasso to features keep:", features_to_keep

# -------------------------------------------------------------------------
# TEST: Raw dataset and Linear feature selection
# -------------------------------------------------------------------------
def test_linear_selected_dataset(train_filename, valid_filename):
    test_name = "normalized and Linear feature selection"
    (all_features, all_targets) = mh.extract(train_filename)

    features_to_keep = selection.extract_linear_features_indexes(all_features, all_targets)
    (sel_features, sel_targets) = selection.extract_features(features_to_keep, all_features, all_targets)

    (valid_features, valid_targets) = mh.extract(valid_filename)
    (sel_valid_features, sel_valid_targets) = selection.extract_features(features_to_keep, valid_features, valid_targets)

    test_classifiers(sel_features, sel_targets, sel_valid_features, sel_valid_targets, test_name)
    print "Linear features to keep:", features_to_keep

def test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name):
    names = [
        # "NB M",
        "NB G",
        "LR",
        # "DT",
        # "KNN",
        # "SVM",
        # "LDA",
        # "QDA",
        # "RFrst",
        "ABoost",
        "Nnet",
        # "ML-LR",
        # "ML-LR2",
        ]
    classifiers = [
        # qc.QuoraMultiNB(all_features, all_targets),
        qc.QuoraGaussianNB(all_features, all_targets),
        qc.QuoraLR(all_features, all_targets),
        # qc.QuoraDT(all_features, all_targets),
        # qc.QuoraKNN(all_features, all_targets),
        # qc.QuoraSVC(all_features, all_targets),
        # qc.QuoraLDA(all_features, all_targets),
        # qc.QuoraQDA(all_features, all_targets),
        # qc.QuoraRandomForest(all_features, all_targets),
        qc.QuoraAdaBoost(all_features, all_targets),
        nnet.QuoraNnet(all_features, all_targets),
        # lr.QuoraMlLR(all_features, all_targets),
        # lrt.QuoraMlLRT(all_features, all_targets),
        ]

    make_section("Test: %s" % test_name)

    for name, clf in zip(names, classifiers):
        start = time()

        clf.train()
        accuracy = clf.accuracy(valid_features, valid_targets)

        # print clf.coef_
        # print clf.coef_.shape

        elapsed = time() - start

        print "%s \t %s \t (%.4f seconds)" % (name, accuracy, elapsed)
    print ""

def make_section(name):
    print "-"*80, "\n", name, "\n", "-"*80

def print_dataset_info():
    make_section("Dataset Information")

    (all_features, all_targets) = mh.extract(train_filename)

    print "Min values in features:", all_features.min(axis=0)
    print "Max values in features:", all_features.max(axis=0)

if __name__ == "__main__":
    train_filename = 'dataset/train.txt'
    valid_filename = 'dataset/valid.txt'

    print_dataset_info()

    test_raw_dataset(train_filename, valid_filename)
    test_selected_dataset(train_filename, valid_filename)
    test_normalized_dataset(train_filename, valid_filename)
    test_normalized_selected_dataset(train_filename, valid_filename)
    test_foba_selected_dataset(train_filename, valid_filename)
    test_random_forest_selected_dataset(train_filename, valid_filename)
    test_lasso_selected_dataset(train_filename, valid_filename)
    # test_linear_selected_dataset(train_filename, valid_filename)
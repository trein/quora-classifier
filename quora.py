import ml_helpers as mh
import quora_classifiers as qc
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
# TEST: Raw dataset and features selection
# -------------------------------------------------------------------------
def test_selected_dataset(train_filename, valid_filename):
    test_name = "raw dataset with feature selection"
    (all_features, all_targets) = mh.extract_selected(train_filename)
    (valid_features, valid_targets) = mh.extract_selected(valid_filename)
    test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Normalized dataset
# -------------------------------------------------------------------------
def test_normalized_dataset(train_filename, valid_filename):
    test_name = "normalized dataset"
    (all_features, all_targets) = mh.extract_normalized(train_filename)
    (valid_features, valid_targets) = mh.extract_normalized(valid_filename)
    test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Normalized dataset and features selection
# -------------------------------------------------------------------------
def test_normalized_selected_dataset(train_filename, valid_filename):
    test_name = "normalized dataset with feature selection"
    (all_features, all_targets) = mh.extract_normalized_selected(train_filename)
    (valid_features, valid_targets) = mh.extract_normalized_selected(valid_filename)
    test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name)

def test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name):
    names = [
        "NB M",
        "NB G",
        "LR",
        "DT",
        "KNN",
        "SVC",
        "LDA",
        "QDA",
        ]
    classifiers = [
        qc.QuoraMultiNB(all_features, all_targets),
        qc.QuoraGaussianNB(all_features, all_targets),
        qc.QuoraLR(all_features, all_targets),
        qc.QuoraDT(all_features, all_targets),
        qc.QuoraKNN(all_features, all_targets),
        qc.QuoraSVC(all_features, all_targets),
        qc.QuoraLDA(all_features, all_targets),
        qc.QuoraQDA(all_features, all_targets),
        ]

    print "-"*80, "\n", "Test: %s" % test_name, "\n", "-"*80
    
    for name, clf in zip(names, classifiers):
        start = time()
        
        clf.train()
        accuracy = clf.accuracy(valid_features, valid_targets)

        elapsed = time() - start

        print "%s \t %s \t (%.4f seconds)" % (name, accuracy, elapsed)
    print ""

if __name__ == "__main__":
    train_filename = 'dataset/train.txt'
    valid_filename = 'dataset/valid.txt'

    test_raw_dataset(train_filename, valid_filename)
    test_selected_dataset(train_filename, valid_filename)
    test_normalized_dataset(train_filename, valid_filename)
    test_normalized_selected_dataset(train_filename, valid_filename)

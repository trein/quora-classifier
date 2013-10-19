import ml_helpers as mh
import quora_classifiers as qc

# -------------------------------------------------------------------------
# TEST: Raw dataset (no modification)
# -------------------------------------------------------------------------
def test_raw_dataset(train_filename, valid_filename):
    test_name = "raw dataset"
    (all_features, all_targets) = mh.extract_all(train_filename)
    (valid_features, valid_targets) = mh.extract_all(valid_filename)
    test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Raw dataset and features selection
# -------------------------------------------------------------------------
def test_selected_dataset(train_filename, valid_filename):
    test_name = "raw dataset with feature selection"
    (all_features, all_targets) = mh.extract_selected_all(train_filename)
    (valid_features, valid_targets) = mh.extract_selected_all(valid_filename)
    test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Normalized dataset
# -------------------------------------------------------------------------
def test_normalized_dataset(train_filename, valid_filename):
    test_name = "normalized dataset"
    (all_features, all_targets) = mh.extract_normalized_all(train_filename)
    (valid_features, valid_targets) = mh.extract_normalized_all(valid_filename)
    test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name)

# -------------------------------------------------------------------------
# TEST: Normalized dataset and features selection
# -------------------------------------------------------------------------
def test_normalized_selected_dataset(train_filename, valid_filename):
    test_name = "normalized dataset with feature selection"
    (all_features, all_targets) = mh.extract_normalized_selected_all(train_filename)
    (valid_features, valid_targets) = mh.extract_normalized_selected_all(valid_filename)
    test_classifiers(all_features, all_targets, valid_features, valid_targets, "")

def test_classifiers(all_features, all_targets, valid_features, valid_targets, test_name):
    names = [
        "NB M",
        "NB G",
        "LR",
        "DT",
        "KNN",
        "SVC",
        "QDA",
        ]
    classifiers = [
        qc.QuoraMultiNB(all_features, all_targets),
        qc.QuoraGaussianNB(all_features, all_targets),
        qc.QuoraLR(all_features, all_targets),
        qc.QuoraDT(all_features, all_targets),
        qc.QuoraKNN(all_features, all_targets),
        qc.QuoraSVC(all_features, all_targets),
        qc.QuoraQDA(all_features, all_targets),
        ]

    print "-"*80, "\n", "Test: %s" % test_name, "\n", "-"*80
    
    for name, clf in zip(names, classifiers):
        clf.train()
        accuracy = clf.accuracy(valid_features, valid_targets)

        print "%s \t %s" % (name, accuracy)
    print ""

if __name__ == "__main__":
    train_filename = 'dataset/train.txt'
    valid_filename = 'dataset/valid.txt'

    test_raw_dataset(train_filename, valid_filename)
    test_selected_dataset(train_filename, valid_filename)
    test_normalized_dataset(train_filename, valid_filename)
    test_normalized_selected_dataset(train_filename, valid_filename)

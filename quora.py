import ml_helpers as mh
import quora_classifiers as qc

# -------------------------------------------------------------------------
# TEST: Raw dataset (no modification)
# -------------------------------------------------------------------------
def test_raw_dataset():
    (all_features, all_targets) = mh.extract_all()

    test_classifiers(all_features, all_targets, "raw dataset")

# -------------------------------------------------------------------------
# TEST: Raw dataset and features selection
# -------------------------------------------------------------------------
def test_selected_dataset():
    (all_features, all_targets) = mh.extract_selected_all()

    test_classifiers(all_features, all_targets, "raw dataset with feature selection")

# -------------------------------------------------------------------------
# TEST: Normalized dataset
# -------------------------------------------------------------------------
def test_normalized_dataset():
    (all_features, all_targets) = mh.extract_normalized_all()

    test_classifiers(all_features, all_targets, "normalized dataset")

# -------------------------------------------------------------------------
# TEST: Normalized dataset and features selection
# -------------------------------------------------------------------------
def test_normalized_selected_dataset():
    (all_features, all_targets) = mh.extract_normalized_selected_all()

    test_classifiers(all_features, all_targets, "normalized dataset with feature selection")

def test_classifiers(all_features, all_targets, test_name):
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
        accuracy = clf.accuracy()

        print "%s \t %s" % (name, accuracy)
    print ""

if __name__ == "__main__":
    test_raw_dataset()
    test_selected_dataset()
    test_normalized_dataset()
    test_normalized_selected_dataset()

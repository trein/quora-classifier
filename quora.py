import ml_helpers as mh
import quora_classifiers as qc

# -------------------------------------------------------------------------
# TEST1: Raw dataset (no modification)
# -------------------------------------------------------------------------
def test_raw_dataset():
    (features, targets) = mh.extract_train()
    (features_test, targets_test) = mh.extract_test()

    nb_accuracy, lr_accuracy = linear_classifiers(features, targets, features_test, targets_test)

    print "NB without normalization", nb_accuracy
    print "LR without normalization", lr_accuracy

# -------------------------------------------------------------------------
# TEST2: Normalized dataset
# -------------------------------------------------------------------------
def test_normalized_dataset():
    (features, targets) = mh.extract_normalized_train()
    (features_test, targets_test) = mh.extract_normalized_test()

    nb_accuracy, lr_accuracy = linear_classifiers(features, targets, features_test, targets_test)

    print "NB without normalization", nb_accuracy
    print "LR without normalization", lr_accuracy

# -------------------------------------------------------------------------
# TEST3: Normalized dataset and features selection
# -------------------------------------------------------------------------
def test_selected_dataset():
    (features, targets) = mh.extract_selected_train()
    (features_test, targets_test) = mh.extract_selected_test()

    nb_accuracy, lr_accuracy = linear_classifiers(features, targets, features_test, targets_test)

    print "NB without normalization", nb_accuracy
    print "LR without normalization", lr_accuracy

def linear_classifiers(features, targets, features_test, targets_test):
    nb = qc.QuoraNB()
    nb.train(features, targets)
    nb_accuracy = nb.accuracy(features_test, targets_test)

    lr = qc.QuoraLR()
    lr.train(features, targets)
    lr_accuracy = lr.accuracy(features_test, targets_test)

    return nb_accuracy, lr_accuracy

if __name__ == "__main__":
    test_raw_dataset()
    test_normalized_dataset()
    test_selected_dataset()

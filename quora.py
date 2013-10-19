import ml_helpers as mh
import quora_classifiers as qc

# -------------------------------------------------------------------------
# TEST1: Raw dataset (no modification)
# -------------------------------------------------------------------------
def test_raw_dataset():
    (features, targets) = mh.extract_train()
    (features_test, targets_test) = mh.extract_test()

    accuracy = classifiers_accuracy(features, targets, features_test, targets_test)

    print "NB Multi without normalization", accuracy["mnb"]
    print "NB Gaussian without normalization", accuracy["gnb"]
    print "LR without normalization", accuracy["lr"]
    print "DT without normalization", accuracy["dt"]
    print "KNN without normalization", accuracy["knn"]
    print "SVC without normalization", accuracy["svc"]
    print "QDA without normalization", accuracy["qda"]
    print

# -------------------------------------------------------------------------
# TEST2: Normalized dataset
# -------------------------------------------------------------------------
def test_normalized_dataset():
    (features, targets) = mh.extract_normalized_train()
    (features_test, targets_test) = mh.extract_normalized_test()

    accuracy = classifiers_accuracy(features, targets, features_test, targets_test)

    print "NB Multi with normalization", accuracy["mnb"]
    print "NB Gaussian with normalization", accuracy["gnb"]
    print "LR with normalization", accuracy["lr"]
    print "DT with normalization", accuracy["dt"]
    print "KNN with normalization", accuracy["knn"]
    print "SVC with normalization", accuracy["svc"]
    print "QDA with normalization", accuracy["qda"]
    print

# -------------------------------------------------------------------------
# TEST3: Normalized dataset and features selection
# -------------------------------------------------------------------------
def test_selected_dataset():
    (features, targets) = mh.extract_selected_train()
    (features_test, targets_test) = mh.extract_selected_test()

    accuracy = classifiers_accuracy(features, targets, features_test, targets_test)

    print "NB Multi with selection", accuracy["mnb"]
    print "NB Gaussian with selection", accuracy["gnb"]
    print "LR with selection", accuracy["lr"]
    print "DT with selection", accuracy["dt"]
    print "KNN with selection", accuracy["knn"]
    print "SVC with selection", accuracy["svc"]
    print "QDA with selection", accuracy["qda"]
    print

def classifiers_accuracy(features, targets, features_test, targets_test):
    mnb = qc.QuoraMultiNB()
    mnb.train(features, targets)
    mnb_accuracy = mnb.accuracy(features_test, targets_test)

    lr = qc.QuoraLR()
    lr.train(features, targets)
    lr_accuracy = lr.accuracy(features_test, targets_test)

    
    dt = qc.QuoraDT()
    dt.train(features, targets)
    dt_accuracy = lr.accuracy(features_test, targets_test)

    gnb = qc.QuoraGaussianNB()
    gnb.train(features, targets)
    gnb_accuracy = gnb.accuracy(features_test, targets_test)

    
    knn = qc.QuoraKNN()
    knn.train(features, targets)
    knn_accuracy = knn.accuracy(features_test, targets_test)

    svc = qc.QuoraSVC()
    svc.train(features, targets)
    svc_accuracy = svc.accuracy(features_test, targets_test)

    qda = qc.QuoraQDA()
    qda.train(features, targets)
    qda_accuracy = qda.accuracy(features_test, targets_test)

    return {
        "mnb" : mnb_accuracy,
        "gnb": gnb_accuracy,
        "lr" : lr_accuracy,
        "dt" : dt_accuracy,
        "knn" : knn_accuracy,
        "svc" : svc_accuracy,
        "qda" : qda_accuracy,
        }

if __name__ == "__main__":
    test_raw_dataset()
    test_normalized_dataset()
    test_selected_dataset()

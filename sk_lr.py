import numpy as np
import pylab as pl
import ml_helpers as mh
from sklearn import linear_model

(features, targets) = mh.extract_train()
(features_test, targets_test) = mh.extract_test()

classifier = linear_model.LogisticRegression(C = 1e5, tol = 0.0001)
classifier.fit(features, targets)

target_test_hat = classifier.predict(features_test)
accuracy = mh.accuracy(target_test_hat, targets_test)

print accuracy
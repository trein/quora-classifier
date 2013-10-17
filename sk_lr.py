import numpy as np
import pylab as pl
import feature_extractor as fe
from sklearn import linear_model

(features, targets) = fe.extract_train()
(features_test, targets_test) = fe.extract_test()

classifier = linear_model.LogisticRegression(C = 1e5, tol = 0.0001)
classifier.fit(features, targets)

target_test_hat = classifier.predict(features_test)

accuracy = (1.0 * (target_test_hat == targets_test)).sum(0) / targets_test.shape

print accuracy
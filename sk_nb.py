import numpy as np
import ml_helpers as mh
from sklearn.naive_bayes import MultinomialNB

(features, targets) = mh.extract_train()
(features_test, targets_test) = mh.extract_test()

# print X.shape
# print y.shape
# print features.shape
# print targets.shape

classifier = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior = True)
classifier.fit(features, targets)

target_test_hat = classifier.predict(features_test)
accuracy = mh.accuracy(target_test_hat, targets_test)

print accuracy
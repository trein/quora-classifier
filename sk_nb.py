import numpy as np
import ml_helpers as mh
from sklearn.naive_bayes import MultinomialNB

(features, targets) = mh.extract_train()
(features_test, targets_test) = mh.extract_test()

# X = np.random.randint(5, size=(6, 100))
# y = np.array([1, 2, 3, 4, 5, 6])
# print X.shape
# print y.shape
# print features
# print targets.shape

classifier = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior = True)
classifier.fit(features, targets)

target_test_hat = classifier.predict(features_test)
accuracy = mh.accuracy(target_test_hat, targets_test)

accuracy = classifier.score(features_test, targets_test)

print "Accuracy:", accuracy
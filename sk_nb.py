import numpy as np
import feature_extractor as fe
from sklearn.naive_bayes import MultinomialNB

# X = np.random.randint(5, size=(6, 100))
# y = np.array([1, 2, 3, 4, 5, 6])

(features, targets) = fe.extract_train()
(features_test, targets_test) = fe.extract_test()

# print X.shape
# print y.shape
# print features.shape
# print targets.shape

classifier = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior = True)
classifier.fit(features, targets)

target_test_hat = classifier.predict(features_test)
accuracy = (1.0 * (target_test_hat == targets_test)).sum(0) / targets_test.shape

print accuracy
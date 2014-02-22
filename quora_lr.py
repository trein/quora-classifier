import quora_classifiers as qc
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals import six
from abc import ABCMeta


class QuoraMlLR(qc.QuoraClassifier):
    """
    Class encapsulating a custom implementation of a Logistic
    Regression classifier.
    """

    def __init__(self, features, targets):
        classifier = LogisticRegression()
        qc.QuoraClassifier.__init__(self, classifier, features, targets)


class LogisticRegression(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    """
    Custom implementation of a Logistic Regression classifier
    done during the first assignment of the course CSC2515.
    """

    def __init__(self, eps=0.05, l2=0.05, iterations=2000):
        self.eps = eps
        self.l2 = l2
        self.iterations = iterations

    def score(self, features, targets):
        """
        Compute the score associated to the provided dataset. It follows the
        protocol imposed by scikit-learn classifiers (BaseEstimator).
        """

        predicted_target = self.predict(features)
        num_samples = targets.shape[0]
        fraction_correct = 1.0 * ((predicted_target>.5)==(targets==1)).sum(0) / num_samples

        return fraction_correct

    def fit(self, train_features, train_targets):
        """
        Train the current classifier.
        """

        # seed the random number generator so results are the same each run.
        np.random.seed(1)
        self.coef_ = np.zeros(train_features.shape[1])

        for t in xrange(self.iterations):
            self.update_weights(train_features, train_targets)

    def predict(self, features):
        """
        Perform prediction of targets given an array of features.
        """

        return self.sigmoid(self.raw_predict(features))

    def raw_predict(self, features):
        """
        Raw prediction for Logistic Regression classifier.
        """

        features_bias = features
        return np.dot(features_bias, self.coef_)

    def update_weights(self, train_features, train_targets):
        """
        Training logic associated to model parameters update.
        """

        predicted_target = self.predict(train_features)
        error = train_targets - predicted_target
        train_features_bias = train_features

        dw = np.dot(train_features_bias.T, error) + 2*self.l2*self.coef_

        self.coef_ += self.eps * dw

    def neg_log_likelihood(self, data, labels):
        """
        Compute the negative likelihood.
        """

        z = self.raw_predict(data)
        ll = labels*z - np.log(1 + np.exp(z))

        return - ll.sum(0) + (self.l2 * (np.linalg.norm(self.coef_)**2))

    def sigmoid(self, a):
        """
        Compute sigmoid function for a given number/array of values.
        """

        return 1.0 / (1.0 + np.exp(-a))
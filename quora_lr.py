import quora_classifiers as qc
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals import six
from abc import ABCMeta

class QuoraMlLR(qc.QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = LogisticRegression()
        qc.QuoraClassifier.__init__(self, classifier, all_features, all_targets)


class LogisticRegression(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, eps=0.05, l2=0.05, iterations=2000):
        self.eps = eps
        self.l2 = l2
        self.iterations = iterations

    def score(self, features, targets):
        predicted_target = self.predict(features)
        num_samples = targets.shape[0]
        fraction_correct = 1.0 * ((predicted_target>.5)==(targets==1)).sum(0) / num_samples

        return fraction_correct

    def fit(self, train_features, train_targets):
        # seed the random number generator so results are the same each run.
        np.random.seed(1)

        # coefficients must consider the bias term
        # self.coef_ = 0.01*np.random.randn(train_features.shape[1])
        # self.coef_ = np.hstack((1, self.coef_))
        self.coef_ = np.zeros(train_features.shape[1])
        # self.coef_ = np.hstack((1, self.coef_))

        # print train_features.shape
        # print train_targets.shape
        # self.update_weights(train_features, train_targets)

        for t in xrange(self.iterations):
            # neg_log_likelihood = self.neg_log_likelihood(train_features, train_targets)
            # print ('ITERATION %4i   LOGL:%4.2f' % (t, neg_log_likelihood))
            # if not np.isfinite(neg_log_likelihood):
            #     raise Exception('nan/inf error')

            self.update_weights(train_features, train_targets)

    def predict(self, features):
        return self.sigmoid(self.raw_predict(features))

    def raw_predict(self, features):
        # features_bias = np.hstack((np.ones((features.shape[0], 1)), features))
        features_bias = features
        return np.dot(features_bias, self.coef_)

    def update_weights(self, train_features, train_targets):
        predicted_target = self.predict(train_features)
        error = train_targets - predicted_target
        # train_features_bias = np.hstack((np.ones((train_features.shape[0], 1)), train_features))
        train_features_bias = train_features

        dw = np.dot(train_features_bias.T, error) + 2*self.l2*self.coef_

        self.coef_ += self.eps * dw

    def neg_log_likelihood(self, data, labels):
        z = self.raw_predict(data)
        ll = labels*z - np.log(1 + np.exp(z))

        return - ll.sum(0) + (self.l2 * (np.linalg.norm(self.coef_)**2))

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))
import quora_classifiers as qc
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals import six
from abc import ABCMeta

class QuoraMlLR(qc.QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = LR()
        qc.QuoraClassifier.__init__(self, classifier, all_features, all_targets)


class LR(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, eps=0.01, l2=0.2, iterations=2000):
        self.eps = eps
        self.l2 = l2
        self.iterations = iterations

    def fit(self, train_features, train_targets):
        np.random.seed(1) #seed the random number generator so results are the same each run.
        self.coef_ = 0.01*np.random.randn(train_features.shape[1])
        self.logistic_err(train_features, train_targets)

        for t in xrange(self.iterations):
            [f, gradient] = self.logistic_err(train_features, train_targets)

            # if not np.isfinite(f):
            #     raise Exception('nan/inf error')

            self.update_weights(gradient)
            # print ('ITERATION %4i   LOGL:%4.2f' % (t, f))

    def score(self, features, targets):
        z = np.dot(features, self.coef_)
        k = self.sigmoid(z)
        num_samples = targets.shape[0]
        fraction_correct = 1.0 * ((k>.5)==(targets==1)).sum(0) / num_samples

        return fraction_correct

    def update_weights(self, gradient):
        self.coef_ += self.eps * gradient

    def logistic_err(self, data, labels):
        """
        Computes the logistic regression negative log-likelihood, gradient,
        and the fraction of correctly predicted cases.

        Given:
            data       - a num_cases by num_dimensions matrix of data examples.
            labels     - a num_cases length vector of binary labels.

        Returns: negative log-likelihood, gradient
        """
        z = np.dot(data, self.coef_)
        ll = labels*z - np.log(1 + np.exp(z))
        k = self.sigmoid(z)

        f = - ll.sum(0) + (self.l2 * (np.linalg.norm(self.coef_)**2))
        df = np.dot(data.T, (labels - k)) + 2*self.l2*self.coef_

        return f, df

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))
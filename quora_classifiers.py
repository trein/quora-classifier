import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model

class QuoraNB:

    def train(self, features, targets):
        self.classifier = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior = True)
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)

class QuoraLR:

    def train(self, features, targets):
        self.classifier = linear_model.LogisticRegression(C = 1e5, tol = 0.0001)
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)
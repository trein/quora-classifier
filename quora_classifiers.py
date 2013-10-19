import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.qda import QDA
from sklearn.svm import SVC

class QuoraMultiNB:

    def train(self, features, targets):
        self.classifier = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior = True)
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)

class QuoraGaussianNB:

    def train(self, features, targets):
        self.classifier = GaussianNB()
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)

class QuoraLR:

    def train(self, features, targets):
        self.classifier = LogisticRegression(C = 1e5, tol = 0.0001)
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)

class QuoraDT:

    def train(self, features, targets):
        self.classifier = DecisionTreeClassifier(min_samples_split = 1, random_state=0)
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)

class QuoraKNN:

    def train(self, features, targets):
        self.classifier = KNeighborsClassifier(n_neighbors = 3)
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)

class QuoraSVC:

    def train(self, features, targets):
        self.classifier = SVC(gamma = 2, C = 1)
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)

class QuoraQDA:

    def train(self, features, targets):
        self.classifier = QDA(reg_param = 0.5)
        self.classifier.fit(features, targets)

    def accuracy(self, features_test, targets_test):
        return self.classifier.score(features_test, targets_test)

class QuoraClassifier:

    def __init__(self, all_features, all_targets):
        self.all_features = all_features
        self.all_targets = all_targets

    def train(self):
        self.classifier.fit(self.train_features(), self.train_targets())

    def accuracy(self):
        return self.classifier.score(self.test_features(), self.test_targets())

    def cross_score(self):
        cross_val_score(self.classifier, self.all_features, self.all_targets, cv=10)

    def train_features(self):
        return self.all_features[0:3500]

    def train_targets(self):
        return self.all_targets[0:3500]

    def test_features(self):
        return self.all_features[3501:4500]

    def test_targets(self):
        return self.all_targets[3501:4500]

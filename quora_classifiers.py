from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.qda import QDA
from sklearn.lda import LDA
from sklearn.svm import SVC

class QuoraClassifier:

    TRAIN_LIMIT = 3500

    def __init__(self, classifier, all_features, all_targets):
        self.classifier = classifier
        self.all_features = all_features
        self.all_targets = all_targets

    def train(self):
        self.classifier.fit(self.train_features(), self.train_targets())
        print self.classifier.coef_

    def accuracy(self, valid_features, valid_targets):
        valid_score = self.valid_score(valid_features, valid_targets)
        test_score = self.test_score()
        cross_score = self.cross_score()
        return "{ valid: %.4f  test: %.4f  cross: %s }" % (valid_score, test_score, cross_score)

    def valid_score(self, valid_features, valid_targets):
        return self.classifier.score(valid_features, valid_targets)

    def test_score(self):
        return self.classifier.score(self.test_features(), self.test_targets())

    def cross_score(self):
        return cross_val_score(self.classifier, self.all_features, self.all_targets, cv=3)

    def train_features(self):
        return self.all_features[0:QuoraClassifier.TRAIN_LIMIT]

    def train_targets(self):
        return self.all_targets[0:QuoraClassifier.TRAIN_LIMIT]

    def test_features(self):
        return self.all_features[QuoraClassifier.TRAIN_LIMIT+1:4500]

    def test_targets(self):
        return self.all_targets[QuoraClassifier.TRAIN_LIMIT+1:4500]

class QuoraMultiNB(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)

class QuoraGaussianNB(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = GaussianNB()
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)

class QuoraLR(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = LogisticRegression(C=1e5, tol=0.001, fit_intercept=True)
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)

class QuoraDT(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = DecisionTreeClassifier(min_samples_split=1, random_state=0)
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)

class QuoraKNN(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = KNeighborsClassifier(n_neighbors=3)
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)

class QuoraSVC(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = SVC(gamma=2, C=1)
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)


class QuoraLDA(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = LDA()
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)

class QuoraQDA(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = QDA(reg_param=0.5)
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)

class QuoraRandomForest(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = RandomForestClassifier(n_estimators=100)
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)

class QuoraAdaBoost(QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = AdaBoostClassifier(n_estimators=100)
        QuoraClassifier.__init__(self, classifier, all_features, all_targets)
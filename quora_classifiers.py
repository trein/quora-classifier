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
    """
    Base abstraction for classifiers behaviors.
    """

    TRAIN_LIMIT = 3500

    def __init__(self, classifier, features, targets):
        self.classifier = classifier
        self.features = features
        self.targets = targets

    def train(self):
        """
        Training logic for the classifier.
        """
        self.classifier.fit(self.train_features(), self.train_targets())

    def accuracy(self, test_features, test_targets):
        """
        Compute the accuracy using validation data as well as
        features/targets from test data.
        """

        cross_score = self.cross_score()
        valid_score = self.valid_score()
        test_score = self.test_score(test_features, test_targets)

        return "{ test: %.4f  valid: %.4f  cross: %s }" % (valid_score, test_score, cross_score)

    def test_score(self, valid_features, valid_targets):
        """
        Compute the accuracy of the classifier considering the test data.
        """

        return self.classifier.score(valid_features, valid_targets)

    def valid_score(self):
        """
        Compute the accuracy of the classifier considering the validation data.
        """

        return self.classifier.score(self.valid_features(), self.valid_targets())

    def cross_score(self):
        """
        Perform three fold cross-validation using the current model.
        """

        return cross_val_score(self.classifier, self.features, self.targets, cv=3)

    def train_features(self):
        """
        Split features array and take features that will be used for training
        the model/classifier.
        """

        return self.features[0:QuoraClassifier.TRAIN_LIMIT]

    def train_targets(self):
        """
        Split targets array and take features that will be used for training
        the model/classifier.
        """

        return self.targets[0:QuoraClassifier.TRAIN_LIMIT]

    def valid_features(self):
        """
        Split features array and take features that will be used for testing
        the generalization capability of the model/classifier.
        """

        return self.features[QuoraClassifier.TRAIN_LIMIT+1:4500]

    def valid_targets(self):
        """
        Split targets array and take features that will be used for testing
        the generalization capability of the model/classifier.
        """

        return self.targets[QuoraClassifier.TRAIN_LIMIT+1:4500]


class QuoraMultiNB(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    Multinomial Naive Bayes classifier.
    """

    def __init__(self, features, targets):
        classifier = MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraGaussianNB(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    Gaussian Naive Bayes classifier.
    """

    def __init__(self, features, targets):
        classifier = GaussianNB()
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraLR(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    Logistic Regression classifier.
    """

    def __init__(self, features, targets):
        classifier = LogisticRegression(C=1e5, tol=0.001, fit_intercept=True)
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraDT(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    Decision Trees classifier.
    """

    def __init__(self, features, targets):
        classifier = DecisionTreeClassifier(min_samples_split=1, random_state=0)
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraKNN(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    K-NN classifier.
    """

    def __init__(self, features, targets):
        classifier = KNeighborsClassifier(n_neighbors=3)
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraSVC(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    SVM classifier.
    """

    def __init__(self, features, targets):
        classifier = SVC(gamma=2, C=1)
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraLDA(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    Linear Discriminant classifier.
    """

    def __init__(self, features, targets):
        classifier = LDA()
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraQDA(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    Quadratic Discriminant classifier.
    """

    def __init__(self, features, targets):
        classifier = QDA(reg_param=0.5)
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraRandomForest(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    Random Forest classifier.
    """

    def __init__(self, features, targets):
        classifier = RandomForestClassifier(n_estimators=200)
        QuoraClassifier.__init__(self, classifier, features, targets)


class QuoraAdaBoost(QuoraClassifier):
    """
    Classifier implementation encapsulating scikit-learn
    AdaBoost classifier.
    """

    def __init__(self, features, targets):
        classifier = AdaBoostClassifier(n_estimators=200)
        QuoraClassifier.__init__(self, classifier, features, targets)
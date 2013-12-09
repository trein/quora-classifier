# Inspired on code provided by Navdeep Jaitly (ndjaitly@gmail.com), 2013
import quora_classifiers as qc
import sys
import numpy as np
from numpy.random import randn
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals import six
from abc import ABCMeta

class QuoraNnet(qc.QuoraClassifier):

    def __init__(self, all_features, all_targets):
        classifier = Brain()
        qc.QuoraClassifier.__init__(self, classifier, all_features, all_targets)


class Brain(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, batch_size=200, eps=0.5, momentum=0.7, l2=0.01, hidden_units=300, num_layers=1, max_epochs=50):
        self.batch_size = batch_size
        self.eps = eps
        self.momentum = momentum
        self.l2 = l2
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.max_epochs = max_epochs

    def fit(self, train_features, train_targets, valid_features=None, valid_targets=None):
        train_features = train_features
        train_targets = self.expand(train_targets)

        # print "train_features", train_features.shape
        # print "train_targets", train_targets.shape

        # Definition of multi layer neural network
        lst_def = self.create_layers_def(train_features, train_targets, self.num_layers, self.hidden_units)
        self.nnet = NNet(lst_def)

        # object encapsulating early stopping logic
        stopper = EarlyStopper()

        for i in range(self.max_epochs):
            self.nnet.train_for_one_epoch(train_features, train_targets, self.eps, self.momentum, self.l2, self.batch_size)

            if self.should_stop_training(i+1, stopper, valid_features, valid_targets):
                break

    def score(self, features, targets):
        validation_accuracy, log_p = self.nnet.test(features.T, self.expand(targets))
        return validation_accuracy

    def expand(self, array):
        return np.hstack((np.vstack((array+1)/2), np.vstack((array-1)/-2)))

    def should_stop_training(self, epoch, stopper, valid_features, valid_targets):
        if not valid_targets or not valid_features:
            return False

        validation_accuracy, log_p = self.nnet.test(valid_features, valid_targets)
        validation_error = 100 - validation_accuracy

        # -------------------------------------------------------------------------
        # early stopping strategy for controlling over-fitting
        # -------------------------------------------------------------------------
        return stopper.should_early_stop(epoch, validation_error)

    def create_layers_def(self, features, targets, num_layers, hidden_units):
        lst_def = []
        in_layer_def = LayerDefinition.new_sigmoid_definition("Layer_in", features.shape[1], hidden_units)
        lst_def.append(in_layer_def)

        for layer_num in range(0, num_layers-1):
            l_name = "Layer_%d" % (layer_num + 1)
            l_type = LayerDefinition.SIGMOID_LAYER
            layer_def = LayerDefinition.new_sigmoid_definition(l_name, l_type, hidden_units, hidden_units)
            lst_def.append(layer_def)

        out_layer_def = LayerDefinition.new_softmax_definition("Layer_out", hidden_units, targets.shape[1])
        lst_def.append(out_layer_def)

        return lst_def


class EarlyStopper:

    def __init__(self):
        self.epochs_to_consider = 3
        self.occurrence = 0
        self.best_epoch = 0
        self.best_validation_error = 100

    def should_early_stop(self, epoch, validation_error):
        should_stop = False
        if validation_error >= self.best_validation_error:
            self.occurrence += 1
            if self.occurrence == self.epochs_to_consider:
                print
                print "Iteration stopped at epoch %d" % epoch
                print "Best validation error found at epoch %d" % self.best_epoch
                print "Best validation error is %f" % self.best_validation_error
                should_stop = True
        else:
            self.occurrence = 0
            self.best_epoch = epoch
            self.best_validation_error = validation_error
        return should_stop


class NNet(object):

    def __init__(self, lst_def):
        self._layers = []
        self.num_layers = len(lst_def)

        self._data_dim = lst_def[0].input_dim

        self._lst_num_hid = []
        self._lst_layer_type = []
        self._lst_layers = []

        for layer_num, layer_def in enumerate(lst_def):
            self._lst_num_hid.append(layer_def.num_units)
            self._lst_layer_type.append(layer_def.layer_type)
            self._lst_layers.append(layer_def.create_layer())

    def get_num_layers(self):
        return len(self._lst_layers)

    def get_code_dim(self):
        return self._lst_num_hid[-1]

    def fwd_prop(self, data):
        lst_layer_outputs = []
        current_layer_output = data

        for layer in self._lst_layers:
            current_layer_output = layer.fwd_prop(current_layer_output)
            lst_layer_outputs.append(current_layer_output)

        return lst_layer_outputs

    def back_prop(self, lst_layer_outputs, data, targets):
        layers_outputs = lst_layer_outputs[::-1]
        layers = self._lst_layers[::-1]

        prev_layers_outputs = lst_layer_outputs[::-1][1:]
        prev_layers_outputs.append(data)

        output_grad = 0

        for layer, layer_outputs, prev_layer_outputs in zip(layers, layers_outputs, prev_layers_outputs):
            if layer.is_softmax():
                act_grad = layer.compute_act_gradients_from_targets(targets, layer_outputs)
                input_grad = layer.back_prop(act_grad, prev_layer_outputs)
                output_grad = input_grad
            else:
                act_grad = layer.compute_act_grad_from_output_grad(layer_outputs, output_grad)
                input_grad = layer.back_prop(act_grad, prev_layer_outputs)
                output_grad = input_grad

    def apply_gradients(self, eps, momentum, l2, batch_size):
        for layer in self._lst_layers:
            layer.apply_gradients(momentum, eps, l2, batch_size)

    def test(self, valid_features, valid_targets):
        """
        Function used to test accuracy.
        """
        num_pts = valid_targets.shape[0]
        lst_layer_outputs = self.fwd_prop(valid_features)

        # print "Data", valid_features.shape
        # print "Target", valid_targets.shape
        # print "Got", lst_layer_outputs[-1]
        # print "Exp", valid_targets.T

        num_correct, log_prob = self._lst_layers[-1].compute_accuraccy(lst_layer_outputs[-1], valid_targets.T)
        classification_error = (num_pts - num_correct)*1.0/num_pts

        return 1 - classification_error, log_prob*1./num_pts

    def train_for_one_epoch(self, features, targets, eps, momentum, l2, batch_size):
        try:
            self.__cur_epoch += 1
        except AttributeError:
            self.__cur_epoch = 1

        try:
            self._tot_batch
        except AttributeError:
            self._tot_batch = 0

        num_pts, classif_err_sum, lg_p_sum = 0, 0, 0
        batch = 0

        for (chuck_feature, chunk_target) in self.get_iterator(features, targets, batch_size):
            batch += 1
            num_pts += batch_size

            # print "Batch chuck_feature", chuck_feature.shape
            # print "Batch chunk_target", chunk_target.shape

            lst_layer_outputs = self.fwd_prop(chuck_feature)

            # print "Got", lst_layer_outputs[-1]
            # print "Exp", chunk_target

            num_correct, log_prob = self._lst_layers[-1].compute_accuraccy(lst_layer_outputs[-1], chunk_target)
            classif_err_sum += (chuck_feature.shape[1] - num_correct)
            lg_p_sum += log_prob

            # print "Batch classif error" classif_err_sum

            self.back_prop(lst_layer_outputs, chuck_feature, chunk_target)
            self.apply_gradients(eps, momentum, l2, batch_size)
            self._tot_batch += 1

        classification_error = classif_err_sum*100./num_pts

        # sys.stderr.write("Epoch = %d, batch = %d, Classif Acc = %.3f, lg(p) %.4f\n"%(\
        #            self.__cur_epoch, batch, 100-classification_error, lg_p_sum*1./num_pts))
        sys.stderr.flush()

        return classification_error

    def get_iterator(self, features, targets, batch_size):
        """
        Yield successive n-sized chunks from l.
        """
        for i in xrange(0, len(targets), batch_size):
            yield features[i:i+batch_size].T, targets[i:i+batch_size].T


class LayerDefinition(object):

    SIGMOID_LAYER = 0
    SOFTMAX_LAYER = 1

    def __init__(self, name, layer_type, input_dim, num_units, wt_sigma):
        self.name, self.layer_type, self.input_dim, self.num_units, \
        self.wt_sigma  =  name, layer_type, input_dim, num_units, wt_sigma

    def create_layer(self):
        layer = None
        if self.layer_type == LayerDefinition.SIGMOID_LAYER:
            layer = SigmoidLayer(self)
        elif self.layer_type == LayerDefinition.SOFTMAX_LAYER:
            layer = SoftmaxLayer(self)
        else:
            raise Exception, "Unknown layer type"
        return layer

    @staticmethod
    def new_softmax_definition(name, input_dim, num_units, wt_sigma=0.01):
        return LayerDefinition(name, LayerDefinition.SOFTMAX_LAYER, input_dim, num_units, wt_sigma)

    @staticmethod
    def new_sigmoid_definition(name, input_dim, num_units, wt_sigma=0.01):
        return LayerDefinition(name, LayerDefinition.SIGMOID_LAYER, input_dim, num_units, wt_sigma)


class Layer(object):

    def __init__(self, layer_def):
        self.name = layer_def.name
        input_dim, output_dim, wt_sigma = layer_def.input_dim, \
                        layer_def.num_units, layer_def.wt_sigma

        self._wts = randn(input_dim, output_dim) * wt_sigma
        self._b = np.zeros((output_dim, 1))

        self._wts_grad = np.zeros(self._wts.shape)
        self._wts_inc = np.zeros(self._wts.shape)

        self._b_grad = np.zeros(self._b.shape)
        self._b_inc = np.zeros(self._b.shape)

        self.__num_params = input_dim*output_dim

    @property
    def shape(self):
        return self._wts.shape

    @property
    def num_hid(self):
        return self._wts.shape[1]

    @property
    def num_dims(self):
        return self._wts.shape[0]

    def apply_gradients(self, momentum, eps, l2, batch_size):
        w_momentum = momentum * self._wts_inc
        b_momentum = momentum * self._b_inc

        w_learning = - (self._wts_grad * eps / batch_size)
        b_learning = - (self._b_grad * eps / batch_size)

        w_l2 = - (l2 * self._wts * eps / batch_size)
        b_l2 = - (l2 * self._b * eps / batch_size)

        self._wts_inc = w_learning + w_l2 + w_momentum
        self._b_inc = b_learning + b_l2 + b_momentum

        self._wts += self._wts_inc
        self._b += self._b_inc

    def back_prop(self, act_grad, prev_layer_outputs):
        self._wts_grad = np.dot(prev_layer_outputs, act_grad.T)
        self._b_grad = act_grad.sum(1)[:, np.newaxis]
        input_grad = np.dot(self._wts, act_grad)

        return input_grad


class SigmoidLayer(Layer):
    pass

    def fwd_prop(self, data):
        # print "Data", data.shape
        # print "Weights", self._wts.shape

        a = np.dot(data.T, self._wts) + self._b.T
        outputs = self.sigmoid(a)

        return outputs.T

    def compute_act_grad_from_output_grad(self, layer_outputs, output_grad):
        act_grad = self.dsigmoid(layer_outputs)

        return act_grad * output_grad

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1.0 - x)

    def is_softmax(self):
        return False


class SoftmaxLayer(Layer):
    pass

    def fwd_prop(self, data):
        a = np.dot(data.T, self._wts) + self._b.T
        outputs = self.softmax(a)

        return outputs.T

    def compute_act_gradients_from_targets(self, targets, output):
        act_grad = output - targets

        return act_grad

    def softmax(self, x):
        exp_x_sum = np.vstack(np.exp(x).sum(1))
        return 1.0 * np.exp(x) / exp_x_sum

    def is_softmax(self):
        return True

    @staticmethod
    def compute_accuraccy(probs, label_mat):
        # print "Gets", probs.argmax(axis=0)
        # print "Expects", label_mat.argmax(axis=0)

        num_correct = np.sum(probs.argmax(axis=0) == label_mat.argmax(axis=0))
        log_probs = np.sum(np.log(probs) * label_mat)

        # print "Logs", log_probs
        return num_correct, log_probs

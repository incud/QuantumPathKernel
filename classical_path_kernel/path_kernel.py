import numpy as np
import abc
import jax
import jax.numpy as jnp
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin


class PredictorGradientGetter(metaclass=abc.ABCMeta):

    def get_epochs_trained(self):
        raise NotImplementedError()

    def get_params(self, epoch):
        raise NotImplementedError()

    def get_predictor_gradient(self, params, the_x):
        raise NotImplementedError()


class GradientKernel:

    def __init__(self, trained_predictor):
        assert isinstance(trained_predictor, PredictorGradientGetter)
        self.trained_predictor = trained_predictor

        @jax.jit
        def get_pathk_feature_map(the_x, sample_frequency):
            # threshold = 0.1
            n_epochs = trained_predictor.get_epochs_trained()
            parameters_history = [trained_predictor.get_params(i * sample_frequency) for i in range(n_epochs // sample_frequency)]
            # parameters_norm = [jnp.linalg.norm(parameters_history[0])] + \
            #                   [jnp.linalg.norm(parameters_history[i] - parameters_history[i - 1]) for i in range(1, n_epochs)]
            # print(jnp.min(parameters_norm), jnp.max(parameters_norm))
            # if threshold is not None:
            #     parameters_history = parameters_history[parameters_norm >= threshold]
            gradient_matrix = [trained_predictor.get_predictor_gradient(params, the_x) for params in parameters_history]
            gradient_vec = jnp.array(jax.tree_flatten(gradient_matrix)[0]).flatten()
            print(gradient_vec.shape)
            return gradient_vec

        self.get_pathk_feature_map = get_pathk_feature_map

    # def get_ntk_feature_map(self, the_x):
    #     init_parameters = self.trained_predictor.get_params(epoch=0)
    #     gradient_vec = self.trained_predictor.get_predictor_gradient(init_parameters, the_x)
    #     return gradient_vec
    #
    # def get_trained_ntk_feature_map(self, the_x):
    #     last_epoch = self.trained_predictor.get_epochs_trained() - 1
    #     last_parameters = self.trained_predictor.get_params(epoch=last_epoch)
    #     gradient_vec = self.trained_predictor.get_predictor_gradient(last_parameters, the_x)
    #     return gradient_vec

    def path_kernel(self, X_test, X_train=None, sample_frequency=50):
        if X_train is None:
            X_train = X_test
        XX1 = jnp.array([self.get_pathk_feature_map(x, sample_frequency) for x in X_test])
        XX2 = jnp.array([self.get_pathk_feature_map(x, sample_frequency) for x in X_train])
        print(XX1.shape, XX2.shape)
        return XX1.dot(XX2.T)



class PathKernelClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, trained_predictor, sample_frequency):
        assert isinstance(trained_predictor, PredictorGradientGetter)

        @jax.jit
        def get_pathk_feature_map(the_x):
            n_epochs = trained_predictor.get_epochs_trained()
            parameters_history = [trained_predictor.get_params(i * sample_frequency) for i in range(n_epochs // sample_frequency)]
            gradient_matrix = [trained_predictor.get_predictor_gradient(params, the_x) for params in parameters_history]
            gradient_vec = jnp.array(jax.tree_flatten(gradient_matrix)[0]).flatten()
            return gradient_vec

        def path_kernel(X_test, X_train=None):
            if X_train is None:
                X_train = X_test
            XX1 = jnp.array([get_pathk_feature_map(x) for x in X_test])
            XX2 = jnp.array([get_pathk_feature_map(x) for x in X_train])
            print(XX1.shape, XX2.shape)
            return XX1.dot(XX2.T)

        self.svm = SVC(kernel=path_kernel)

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)

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

    def __init__(self, trained_predictor, sample_frequency=None):
        assert isinstance(trained_predictor, PredictorGradientGetter)
        self.trained_predictor = trained_predictor
        self.sample_frequency = self.trained_predictor.get_epochs_trained() // 100 if sample_frequency is None else sample_frequency

        @jax.jit
        def get_pathk_feature_map(the_x):
            n_epochs = trained_predictor.get_epochs_trained()
            parameters_history = [trained_predictor.get_params(i * self.sample_frequency) for i in
                                  range(n_epochs // self.sample_frequency)]
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

        @jax.jit
        def get_gradient_feature_map(the_x, epoch):
            gradient_matrix = trained_predictor.get_predictor_gradient(trained_predictor.get_params(epoch), the_x)
            gradient_vec = jnp.array(jax.tree_flatten(gradient_matrix)[0]).flatten()
            return gradient_vec

        def gradient_kernel(X_test, X_train=None, epoch=0):
            if X_train is None:
                X_train = X_test
            XX1 = jnp.array([get_gradient_feature_map(x) for x in X_test])
            XX2 = jnp.array([get_gradient_feature_map(x) for x in X_train])
            print(XX1.shape, XX2.shape)
            return XX1.dot(XX2.T)

        self.get_pathk_feature_map = get_pathk_feature_map
        self.path_kernel = path_kernel
        self.get_gradient_feature_map = get_gradient_feature_map
        self.gradient_kernel = gradient_kernel

    def set_sample_frequency(self, sample_frequency):
        self.sample_frequency = sample_frequency

    def path_kernel_cached(self, X_test, X_train=None, cache=None):
        if X_train is None:
            X_train = X_test
        epochs = self.trained_predictor.get_epochs_trained()
        for i in range(epochs // self.sample_frequency):
            epoch = i * self.sample_frequency
            if cache[epoch] is None:
                cache[epoch] = self.gradient_kernel(X_test, X_train, epoch=epoch)
        return np.average([cache[i] for i in range(epochs) if cache[i] is not None], axis = 0)

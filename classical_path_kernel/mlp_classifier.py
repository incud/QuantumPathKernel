# Added to silence some warnings.
from jax.config import config
config.update("jax_enable_x64", False)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.base import BaseEstimator, ClassifierMixin
from path_kernel import PredictorGradientGetter

def zero_grads():
    def init_fn(_):
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


class MultiLayerPerceptronClassifier(BaseEstimator, ClassifierMixin, PredictorGradientGetter):

    def __init__(self, hidden_nodes, num_classes, optimizer, epochs, seed, masked_parameters=None, is_linear=False):
        super(BaseEstimator, self).__init__()
        super(ClassifierMixin, self).__init__()
        super(PredictorGradientGetter, self).__init__()
        self.num_classes = num_classes
        self.hidden_nodes = hidden_nodes
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed=seed)
        if not is_linear:
            self.network_fn = lambda x: hk.nets.MLP(self.hidden_nodes + [self.num_classes])(x)
        else:
            self.network_fn = lambda x: hk.Sequential(
                [hk.Flatten()] + [hk.Linear(n) for n in hidden_nodes] + [hk.Linear(self.num_classes)]
            )(x)
        self.network = hk.without_apply_rng(hk.transform(self.network_fn))
        self.optimizer = optimizer
        self.epochs = epochs
        self.initial_params = None
        self.initial_opt_state = None
        self.history_params = []
        self.params = None
        self.opt_state = None
        self.masked_parameters = masked_parameters
        self.is_linear = is_linear
        self.trained = False

        def cross_entropy_loss(params, X_batch, y_batch):
            batch_size, _ = X_batch.shape
            logits = self.network.apply(params, X_batch)
            labels = jax.nn.one_hot(y_batch, self.num_classes)
            l2_regulariser = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
            log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))
            return -log_likelihood / batch_size + 1e-4 * l2_regulariser

        self.cross_entropy_loss = cross_entropy_loss

        @jax.jit
        def step(params, opt_state, the_X_batch, the_y_batch):
            loss, grads = jax.value_and_grad(self.cross_entropy_loss)(params, the_X_batch, the_y_batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        self.step = step

        @jax.jit
        def evaluate(params, the_X, the_y):
            loss = cross_entropy_loss(params, the_X, the_y)
            return loss

        self.evaluate = evaluate

        @jax.jit
        def predict_sign(params, the_X):
            onehot_prediction = jax.nn.softmax(self.network.apply(params, the_X))
            onebit_prediction = jnp.sign(onehot_prediction[:, 1] - onehot_prediction[:, 0])
            onebit_prediction_normalized = (onebit_prediction + 1) / 2
            return onebit_prediction_normalized

        @jax.jit
        def predict_raw(params, the_X):
            onehot_prediction = jax.nn.softmax(self.network.apply(params, the_X))
            onebit_prediction = onehot_prediction[:, 1] - onehot_prediction[:, 0]
            onebit_prediction_normalized = (onebit_prediction + 1) / 2
            return onebit_prediction_normalized

        @jax.jit
        def predict_raw_item(params, the_single_x):
            onehot_prediction = jax.nn.softmax(self.network.apply(params, the_single_x))
            onebit_prediction = onehot_prediction[1] - onehot_prediction[0]
            onebit_prediction_normalized = (onebit_prediction + 1) / 2
            return onebit_prediction_normalized

        self.predict_sign = predict_sign

        self.predict_raw = predict_raw

        self.predict_raw_item = predict_raw_item

    def fit(self, X, y):
        self.initial_params = self.network.init(self.rng, X[0])
        self.initial_opt_state = self.optimizer.init(self.initial_params)
        self.params = self.initial_params
        self.opt_state = self.initial_opt_state
        self.history_params.append(self.params)

        for epoch in range(self.epochs):
            params, opt_state = self.step(self.params, self.opt_state, X, y)
            self.params = params
            self.opt_state = opt_state
            loss = self.evaluate(self.params, X, y)
            print(f"Epoch {epoch:4d} loss {loss}")
            self.history_params.append(self.params)

        self.trained = True

    def predict(self, X):
        assert self.trained
        return self.predict_sign(self.params, X)

    def score(self, X, y, sample_weight=None):
        assert self.trained
        y_pred = self.predict(X)
        y_pred_np = np.array(y_pred)
        y_true_np = np.array(y)
        return np.average(y_true_np == y_pred_np)

    def get_num_parameters(self):
        count = 0
        for param_name, param_group in self.params.items():
            if self.masked_parameters is None or param_name not in self.masked_parameters:
                params_count = sum(jax.tree_map(lambda elem: jnp.prod(jnp.array(elem.shape)), jax.tree_leaves(self.params[param_name])))
                count += params_count
        return count

    def get_epochs_trained(self):
        return len(self.history_params)

    def get_params(self, epoch):
        assert epoch < self.get_epochs_trained()
        return self.history_params[epoch]

    def get_predictor_gradient(self, params, the_x):
        grads_dict = jax.grad(self.predict_raw_item)(params, the_x)
        if self.masked_parameters is not None:
            grads_dict = {k: v for k, v in grads_dict.items() if k not in self.masked_parameters}
        grads_list = [vec.flatten() for vec in jax.tree_flatten(grads_dict)[0]]
        grads = jnp.concatenate(grads_list)
        return grads


class RandomKernelClassifier(MultiLayerPerceptronClassifier):

    def __init__(self, n_hidden_nodes, num_classes, optimizer, epochs, seed):
        hidden_nodes = [n_hidden_nodes]
        masked_optimizer = optax.multi_transform(
            {"trainable": optimizer, "frozen": zero_grads()},
            {"mlp/~/linear_0": "frozen", "mlp/~/linear_1": "trainable"}
        )
        super().__init__(hidden_nodes, num_classes, masked_optimizer, epochs, seed, masked_parameters=["mlp/~/linear_0"])


class NeuralNetworkClassifier(MultiLayerPerceptronClassifier):

    def __init__(self, n_hidden_nodes, num_classes, optimizer, epochs, seed):
        hidden_nodes = [n_hidden_nodes]
        super().__init__(hidden_nodes, num_classes, optimizer, epochs, seed)


class DeepLinearClassifier(MultiLayerPerceptronClassifier):

    def __init__(self, hidden_nodes, num_classes, optimizer, epochs, seed):
        super().__init__(hidden_nodes, num_classes, optimizer, epochs, seed, is_linear=True)


class DeepNonLinearClassifier(MultiLayerPerceptronClassifier):

    def __init__(self, hidden_nodes, num_classes, optimizer, epochs, seed):
        super().__init__(hidden_nodes, num_classes, optimizer, epochs, seed, is_linear=False)


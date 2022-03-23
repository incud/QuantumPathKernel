import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import pandas as pd
from functools import partial
import json
from datetime import datetime


class PennylaneLinearTeacher:

    def __init__(self, n_layers, linear_w):

        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=1)
        self.rng = jax.random.PRNGKey(593539)

        @jax.jit
        @qml.qnode(self.device, interface="jax")
        def one_qubit_learner(x, theta):
            qml.RY(x, wires=0)
            for i, thetai in enumerate(theta):
                if i % 2 == 0:
                    qml.RZ(thetai, wires=0)
                else:
                    qml.RX(thetai, wires=0)
            return qml.expval(qml.PauliZ(0))

        self.linear_w = linear_w
        self.one_qubit_learner = one_qubit_learner

    def generate_dataset(self, size, noise=0):
        X = jax.random.uniform(self.rng, minval=0, maxval=1, shape=(size,))
        Y = X * self.linear_w + jax.random.uniform(self.rng, minval=-noise, maxval=+noise, shape=(size,))
        return X, Y

    def evaluate(self, initial_params, epochs, X_train, Y_train, X_test, Y_test):

        specs = {'initial_params': str(initial_params),
                 'optimizer': 'jax_gradient_flow',
                 'epochs': epochs,
                 'layers': self.n_layers,
                 'linear_w': self.linear_w,
                 'X_train': str(X_train),
                 'Y_train': str(Y_train),
                 'X_test': str(X_test),
                 'Y_test': str(X_test)}

        df = pd.DataFrame(columns=['epoch', 'training_loss', 'testing_loss', 'params'])

        def get_mse_loss(X, Y, params):
            cost = [(self.one_qubit_learner(X[index], params) - Y[index])**2 for index in range(X.shape[0])]
            jax_cost = jnp.array(cost)
            jax_sum = jnp.sum(jax_cost)
            return jax_sum

        training_loss = partial(get_mse_loss, X_train, Y_train)
        training_loss_grad = jax.grad(training_loss)
        testing_loss = partial(get_mse_loss, X_test, Y_test)
        params = initial_params

        for i in range(epochs+1):

            if i > 0:
                params -= training_loss_grad(params)

            df.loc[len(df)] = {
                'epoch': i,
                'training_loss': training_loss(params),
                'testing_loss': testing_loss(params),
                'params': params
            }

        return specs, df


def load_trace(version=1, layers=1):
    return pd.read_pickle(f"linear-teacher-experiments-{version}/trace_{layers}.pickle")


def load_all_traces():
    return [[load_trace(version=version, layers=layers) for layers in range(1, 60+1)] for version in range(1, 20+1)]


def s2np(s):
    return np.array(s[1:-1].split(), dtype=float)


def create_ntk_gram_matrix(specs, traces):

    layers = int(specs["layers"])
    linear_w = float(specs["linear_w"])
    X_train, Y_train, X_test, Y_test = s2np(specs["X_train"]), s2np(specs["Y_train"]), s2np(specs["X_test"]), s2np(specs["Y_test"])
    N_TRAIN = X_train.shape[0]
    N_TEST = X_test.shape[0]
    EPOCHS = 1001

    plt = PennylaneLinearTeacher(n_layers=layers, linear_w=linear_w)
    circuit = plt.one_qubit_learner
    circuit_grad = jax.grad(circuit, argnums=(1,))

    def ntk(x1, x2, params):
        a = jnp.array(circuit_grad(x1, params))
        b = jnp.array(circuit_grad(x2, params))
        return float(a.dot(b.T))

    ntk_training_gram_matrix = np.zeros(shape=(N_TRAIN, N_TRAIN, EPOCHS))
    for e in range(EPOCHS):
        if e % 100 == 0:
            print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} [layers={layers}]: Running training epoch {e}")
        params = traces.loc[e]["params"]
        for i1, x1 in enumerate(X_train):
            for i2, x2 in enumerate(X_train):
                k = ntk(x1, x2, params)
                ntk_training_gram_matrix[i1][i2][e] = k
        np.save(f"ntk_training_gram_{layers}.npy", ntk_training_gram_matrix)

    ntk_testing_gram_matrix = np.zeros(shape=(N_TRAIN, N_TEST, EPOCHS))
    for e in range(EPOCHS):
        if e % 100 == 0:
            print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} [layers={layers}]: Running testing epoch {e}")
        params = traces.loc[e]["params"]
        for i1, x1 in enumerate(X_train):
            for i2, x2 in enumerate(X_test):
                k = ntk(x1, x2, params)
                ntk_testing_gram_matrix[i1][i2][e] = k
        np.save(f"ntk_testing_gram_{layers}.npy", ntk_testing_gram_matrix)


for version in range(1, 15+1):
    for layers in range(1, 60+1):

        if version == 1 and layers <= 28:
            continue

        specs_v = json.load(open(f"linear-teacher-experiments-{version}/specs_{layers}.json"))
        trace_v = load_trace(version=version, layers=layers)
        print(f"============ Starting {version} | {layers} =============")
        create_ntk_gram_matrix(specs_v, trace_v)

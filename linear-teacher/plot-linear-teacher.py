import numpy as np
import pandas.core.frame
import pennylane as qml
from pennylane import numpy as pnp  # -> substituted by JAX
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import json


class PennylaneLinearTeacher:

    def __init__(self, n_layers, linear_w):

        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=1)

        @qml.qnode(self.device)
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
        X = pnp.random.uniform(low=0, high=1, size=(size,))
        Y = X * self.linear_w + pnp.random.uniform(low=-noise, high=+noise, size=(size,))
        return X, Y

    def evaluate(self, initial_params, optimizer, epochs, X_train, Y_train, X_test, Y_test):

        specs = {'initial_params': str(initial_params),
                 'optimizer': str(type(optimizer)),
                 'epochs': epochs,
                 'layers': self.n_layers,
                 'linear_w': self.linear_w,
                 'X_train': str(X_train),
                 'Y_train': str(Y_train),
                 'X_test': str(X_test),
                 'Y_test': str(X_test)}

        df = pd.DataFrame(columns=['epoch', 'training_loss', 'testing_loss', 'params'])

        def get_mse_loss(X, Y, params):
            return sum((self.one_qubit_learner(X[i], params) - Y[i])**2 for i in range(X.shape[0]))

        training_loss = partial(get_mse_loss, X_train, Y_train)
        testing_loss = partial(get_mse_loss, X_test, Y_test)
        params = initial_params

        for i in range(epochs+1):

            if i > 0: params = optimizer.step(training_loss, params)

            df.loc[len(df)] = {
                'epoch': i,
                'training_loss': training_loss(params),
                'testing_loss': testing_loss(params),
                'params': params
            }

        return specs, df


TRAINING_SET_SIZE = 6
TESTING_SET_SIZE = 20
MAX_DEPTH = 60
LINEAR_W = 0.66
MAX_EPOCHS = 1000

training_losses = []
testing_losses = []

def load_trace(i, v=1):
    return pd.read_pickle(f"linear-teacher-experiments-{v}/trace_{i}.pickle")

def print_plot(tr_losses, te_losses, xlog, vaxis=None):
    assert len(tr_losses) == len(te_losses)
    plt.plot(range(len(tr_losses)), tr_losses, label="training loss")
    plt.plot(range(len(tr_losses)), te_losses, label="testing loss")
    plt.legend()
    if xlog: plt.xscale('log')
    if vaxis is not None: plt.axvline(x=vaxis, color='r')
    plt.show()

def print_plot_depth(dfs, depth, xlog=False):
    print_plot(dfs[depth].training_loss, dfs[depth].testing_loss, xlog)

def get_training_loss(df, i):
    if type(i) == list:
        return sum(df.loc[ii].training_loss for ii in i) / len(i)
    else:
        return df.loc[i].training_loss

def get_testing_loss(df, i):
    if type(i) == list:
        return sum(df.loc[ii].testing_loss for ii in i) / len(i)
    else:
        return df.loc[i].testing_loss

def print_plot_epoch(dfs, i, xlog=False, vaxis=None):
    tr_losses = []
    te_losses = []
    for df in dfs:
        tr_losses.append(get_training_loss(df, i))
        te_losses.append(get_testing_loss(df, i))
    print_plot(tr_losses, te_losses, xlog, vaxis)

def create_heatmap(dfs):
    train_loss_hm = pnp.zeros((len(dfs), len(dfs[0].training_loss)))
    test_loss_hm = pnp.zeros((len(dfs), len(dfs[0].training_loss)))
    for i, df in enumerate(dfs):
        train_loss_hm[i] = df.training_loss.to_numpy()
        test_loss_hm[i] = df.testing_loss.to_numpy()
    return train_loss_hm, test_loss_hm


def print_heatmap_log(dfs):

    if type(dfs[0]) == pandas.core.frame.DataFrame:
        training_loss_hm, testing_loss_hm = create_heatmap(dfs)
    else:
        N = len(dfs)
        training_loss_hm, testing_loss_hm = create_heatmap(dfs[0])
        for i in range(1, N):
            a, b = create_heatmap(dfs[i])
            training_loss_hm += a
            testing_loss_hm += b
        training_loss_hm /= N
        testing_loss_hm /= N

    im = plt.pcolor(testing_loss_hm.T)
    plt.colorbar(im, orientation='horizontal')
    plt.yscale('log')
    plt.ylim((1, 1000))
    plt.axvline(x=4, color='r')
    plt.show()

dfs1 = [load_trace(i + 1, v=1) for i in range(MAX_DEPTH)]
dfs2 = [load_trace(i + 1, v=2) for i in range(MAX_DEPTH)]

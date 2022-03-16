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

def main():
    for l in range(MAX_DEPTH):
        l = l + 1
        plteacher = PennylaneLinearTeacher(l, LINEAR_W)
        X_train, Y_train = plteacher.generate_dataset(TRAINING_SET_SIZE, noise=0.1)
        X_test, Y_test = plteacher.generate_dataset(TESTING_SET_SIZE, noise=0)
        initial_params = pnp.random.uniform(low=-0.01, high=0.01, size=(l,))
        optimizer = AdamOptimizer()
        specs, trace = plteacher.evaluate(initial_params, optimizer, MAX_EPOCHS, X_train, Y_train, X_test, Y_test)
        json.dump(specs, open(f"linear-teacher-experiments/specs_{l}.json", "w"))
        trace.to_pickle(f"linear-teacher-experiments/trace_{l}.pickle")
        training_loss = trace.loc[len(trace)-1].training_loss
        testing_loss = trace.loc[len(trace) - 1].testing_loss
        print(f"Training loss is: {training_loss:3.3f}; Testing loss is: {testing_loss:3.3f}")
        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

if __name__ == '__main__':
    main()

# def load_trace(i):
#     return pd.read_pickle(f"linear-teacher-experiments/trace_{i}.pickle")
#
#
# dfs = [load_trace(i+1) for i in range(MAX_DEPTH)]
#
#
# def print_plot(tr_losses, te_losses, xlog):
#     assert len(tr_losses) == len(te_losses)
#     plt.plot(range(len(tr_losses)), tr_losses, label="training loss")
#     plt.plot(range(len(tr_losses)), te_losses, label="testing loss")
#     plt.legend()
#     if xlog: plt.xscale('log')
#     plt.show()
#
# def print_plot_depth(depth, xlog=False):
#     print_plot(dfs[depth].training_loss, dfs[depth].testing_loss, xlog)
#
# def print_plot_epoch(i, xlog=False):
#     tr_losses = []
#     te_losses = []
#     for df in dfs:
#         tr_losses.append(df.loc[i].training_loss)
#         te_losses.append(df.loc[i].testing_loss)
#     print_plot(tr_losses, te_losses, xlog)
#
# def print_heatmap():
#     train_loss_hm = pnp.zeros((len(dfs), len(dfs[0].training_loss)))
#     test_loss_hm = pnp.zeros((len(dfs), len(dfs[0].training_loss)))
#     for i, df in enumerate(dfs):
#         train_loss_hm[i] = df.training_loss.to_numpy()
#         test_loss_hm[i] = df.testing_loss.to_numpy()
#     plt.imshow(train_loss_hm, cmap='hot', interpolation='nearest')
#     plt.show()
#     plt.imshow(test_loss_hm, cmap='hot', interpolation='nearest')
#     plt.show()
#
#
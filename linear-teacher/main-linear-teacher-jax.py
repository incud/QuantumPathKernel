import pennylane as qml
import jax
import jax.numpy as jnp
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


TRAINING_SET_SIZE = 6
TESTING_SET_SIZE = 20
MAX_DEPTH = 60
LINEAR_W = 0.66
MAX_EPOCHS = 1000

training_losses = []
testing_losses = []

def main():
    for v in range(21, 21+1):
        for l in range(MAX_DEPTH):
            l = l + 1
            plteacher = PennylaneLinearTeacher(l, LINEAR_W)
            X_train, Y_train = plteacher.generate_dataset(TRAINING_SET_SIZE, noise=0.1)
            X_test, Y_test = plteacher.generate_dataset(TESTING_SET_SIZE, noise=0)
            initial_params = jax.random.uniform(plteacher.rng, minval=-0.01, maxval=0.01, shape=(l,))
            specs, trace = plteacher.evaluate(initial_params, MAX_EPOCHS, X_train, Y_train, X_test, Y_test)
            json.dump(specs, open(f"linear-teacher-experiments-{v}/specs_{l}.json", "w"))
            trace.to_pickle(f"linear-teacher-experiments-{v}/trace_{l}.pickle")
            training_loss = trace.loc[len(trace)-1].training_loss
            testing_loss = trace.loc[len(trace) - 1].testing_loss
            print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} {v}: Training loss is: {training_loss:3.3f}; Testing loss is: {testing_loss:3.3f}")
            training_losses.append(training_loss)
            testing_losses.append(testing_loss)


if __name__ == '__main__':
    main()
    pass

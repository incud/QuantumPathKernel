import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from pennylane.kernels import kernel_matrix
import pandas as pd
import json
from pathlib import Path
import click


def create_gaussian_mixtures(D, snr, N):
    """
    Create the Gaussian mixture dataset
    :param D: number of dimensions: (x1, x2, 0, .., 0) in R^D
    :param snr: signal to noise ratio
    :param N: number of elements to generate
    :return: dataset
    """
    if N % 4 != 0:
        raise ValueError("The number of elements within the dataset must be a multiple of 4")
    if D < 2:
        raise ValueError("The number of dimensions must be at least 2")
    if snr < 0:
        raise ValueError("Signal to noise ratio must be > 0")

    X = np.zeros((N, D))
    Y = np.zeros((N,))
    centroids = np.array([(.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5)])
    for i in range(N):
        quadrant = i % 4
        Y[i] = 1 if quadrant % 2 == 0 else -1
        X[i][0], X[i][1] = centroids[quadrant] + np.random.uniform(-snr, snr, size=(2,))
    return X, Y


def plot_dataset(X, Y):
    X1 = X[Y == 1]
    X2 = X[Y == -1]
    centroids = np.array([(.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5)])
    plt.title(f"Gaussian Mixtures dataset plot")
    plt.scatter(X1[:, 0].tolist(), X1[:, 1].tolist(), label="First class", color='green')
    plt.scatter(X2[:, 0].tolist(), X2[:, 1].tolist(), label="Second class", color='blue')
    plt.scatter(centroids[:, 0].tolist(), centroids[:, 1].tolist(), label="Centroids", color='black', marker='x')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


def create_qnn(N, layers):
    device = qml.device("default.qubit.jax", wires=N)

    @jax.jit
    @qml.qnode(device, interface='jax')
    def qnn(x, theta):
        # data encoding
        for i in range(N):
            qml.RY(x[i], wires=i)
        # variational form (no BP due to LaRocca et al '21)
        for l in range(layers):
            for j in range(N):
                qml.MultiRZ(theta[l*2], wires=(j, (j+1)%N))
            for j in range(N):
                qml.RX(theta[l*2+1], wires=j)
        # measurement - TODO do we need to change into an Hermitian form? H_{TFIM}
        return qml.expval(qml.PauliZ(0))

    return qnn


def train_qnn(X, Y, qnn, n_params, epochs):
    N, _ = X.shape
    seed = int(datetime.now().strftime('%Y%m%d%H%M%S'))
    rng = jax.random.PRNGKey(seed)
    optimizer = optax.adam(learning_rate=0.1)
    params = jax.random.normal(rng, shape=(n_params,))
    opt_state = optimizer.init(params)

    def calculate_cost_item(x, y, qnn, params):
        value = qnn(x, params)  #.block_until_ready()
        return (value - y) ** 2

    def calculate_cost(X, Y, qnn, params):
        the_cost = 0.0
        for i in range(N):
            the_cost += calculate_cost_item(X[i], Y[i], qnn, params)
        return the_cost

    specs = {'initial_params': str(params),
             'optimizer': 'optax.adam(learning_rate=0.1)',
             'epochs': epochs,
             'n_params': n_params,
             'circuit': 'create_rzz_rx_qnn',
             'seed': seed,
             'X': str(X),
             'Y': str(Y)}

    df = pd.DataFrame(columns=['epoch', 'loss', 'params'])
    df.loc[len(df)] = {
        'epoch': 0,
        'loss': calculate_cost(X, Y, qnn, params),
        'params': params
    }

    for epoch in range(1, epochs+1):
        cost, grad_circuit = jax.value_and_grad(lambda w: calculate_cost(X, Y, qnn, w))(params)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        params = optax.apply_updates(params, updates)
        df.loc[len(df)] = {
            'epoch': epoch,
            'loss': cost,
            'params': params
        }
        if epoch % 50 == 0:
            print(".", end="", flush=True)

    print("")
    return specs, df


def calculate_ntk(X, qnn, df):

    qnn_grad = jax.grad(qnn, argnums=(1,))

    def ntk(x1, x2, params):
        a = jnp.array(qnn_grad(x1, params))
        b = jnp.array(qnn_grad(x2, params))
        return float(a.dot(b.T))

    MIN_NORM_CHANGE = 0.1
    ntk_grams = []
    ntk_gram_indexes = []
    ntk_gram_params = []
    for i, row in df.iterrows():
        params = row["params"]
        if len(ntk_gram_params) == 0 or np.linalg.norm(ntk_gram_params[-1] - params) >= MIN_NORM_CHANGE:
            ntk_gram = kernel_matrix(X, X, kernel=lambda x1, x2: ntk(x1, x2, params))
            ntk_grams.append(ntk_gram)
            ntk_gram_indexes.append(i)
            ntk_gram_params.append(params)

    return ntk_grams, ntk_gram_indexes


def calculate_pk(ntk_grams):
    return np.average(ntk_grams, axis=0)


def calculate_tk_alignment(K1, K2, centered=False):
    if centered:
        means = K1.mean(axis=0)
        K1 -= means[None, :]
        K1 -= means[:, None]
        K1 += means.mean()
        means = K2.mean(axis=0)
        K2 -= means[None, :]
        K2 -= means[:, None]
        K2 += means.mean()
    return np.sum(K1 * K2) / np.linalg.norm(K1) / np.linalg.norm(K2)


def run_qnn(X, Y, layers, epochs):

    N, D = X.shape
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Creating QNN")
    qnn = create_qnn(D, layers)
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Start training")
    specs, df = train_qnn(X, Y, qnn, n_params=2*layers, epochs=epochs)
    specs["layers"] = layers
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Start NTK PK calculation")
    ntk_grams, ntk_gram_indexes = calculate_ntk(X, qnn, df)
    pk_gram = calculate_pk(ntk_grams)
    return specs, df, ntk_grams, ntk_gram_indexes, pk_gram


def run_qnns(D, snr, N, MAX_LAYERS, MAX_EPOCHS):

    X, Y = create_gaussian_mixtures(D, snr, N)
    directory = f"experiment_snr{snr:0.2f}_d{D}_{datetime.now().strftime('%Y%m%d%H%M')}"
    Path(directory).mkdir(parents=True, exist_ok=True)

    for layers in range(1, MAX_LAYERS+1):
        specs, df, ntk_grams, ntk_gram_indexes, pk_gram = run_qnn(X, Y, layers=layers, epochs=MAX_EPOCHS)
        specs["D"] = D
        specs["snr"] = snr
        specs["N"] = N
        json.dump(specs, open(f"{directory}/specs_{layers}.json", "w"))
        df.to_pickle(f"{directory}/trace_{layers}.pickle")
        np.save(f"{directory}/ntk_grams_{layers}.npy", ntk_grams)
        np.save(f"{directory}/ntk_gram_indexes_{layers}.npy", ntk_gram_indexes)
        np.save(f"{directory}/pk_gram_{layers}.npy", pk_gram)


# ========================================================================================
# ====================================== CLI =============================================
# ========================================================================================

@click.group()
def main():
    print("Welcome")
    pass

@main.command()
@click.option('--d', default=2, type=int)
@click.option('--snr', default=0.1, type=float)
@click.option('--n', default=16, type=int)
@click.option('--layers', default=20, type=int)
@click.option('--epochs', default=1000, type=int)
def experiment(d, snr, n, layers, epochs):
    """
    Start the experiments
    :param D: dimensionality of the data (at least 2
    :param snr: signal to noise ratio
    :param N: number of training samples (must be multiple of 4, suggested and default 16)
    :param layers: maximum number of layers (default 20)
    :param epochs: maximum number of training epochs (default 1000)
    :return: nothing, everything is saved to file
    """
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Experiment D={d}, snr={snr}, N={n}, MAX_LAYERS={layers}, MAX_EPOCHS={epochs}")
    run_qnns(d, snr, n, MAX_LAYERS=layers, MAX_EPOCHS=epochs)


@main.command()
@click.option('--directory', type=click.Path(exists=True))
def analyze(directory):
    """
    Analyze the data contained in the given directory
    :param directory: where the experiment data is saved
    :return: nothing, everything is saved to file
    """
    raise ValueError("Not implemented yet")


if __name__ == '__main__':
    main()

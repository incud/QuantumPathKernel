import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import re
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from pennylane.kernels import kernel_matrix
import pandas as pd
import json
from pathlib import Path
import click
from sklearn.svm import SVC


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
        Y[i] = 1 if quadrant % 2 == 0 else -1  # labels are 0 or 1
        X[i][0], X[i][1] = centroids[quadrant] + np.random.uniform(-snr, snr, size=(2,))
    return X, Y


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


def train_qnn(X, Y, qnn, loss, n_params, epochs):
    N, _ = X.shape
    seed = int(datetime.now().strftime('%Y%m%d%H%M%S'))
    rng = jax.random.PRNGKey(seed)
    optimizer = optax.adam(learning_rate=0.1)
    params = jax.random.normal(rng, shape=(n_params,))
    opt_state = optimizer.init(params)

    def calculate_mse_cost(X, Y, qnn, params):
        the_cost = 0.0
        for i in range(N):
            x, y = X[i], Y[i]
            yp = qnn(x, params)
            the_cost += (y - yp)**2
        return the_cost

    def calculate_bce_cost(X, Y, qnn, params):
        the_cost = 0.0
        epsilon = 1e-6
        for i in range(N):
            x, y = X[i], Y[i]
            y = (y + 1)/2 + epsilon  # 1 label -> 1; - label -> 0
            yp = (qnn(x, params) + 1)/2 + epsilon  # 1 label -> 1; - label -> 0
            the_cost += y * jnp.log2(yp) + (1 - y) * jnp.log2(1 - yp)
        return the_cost * (-1/N)

    calculate_cost = calculate_mse_cost if loss == "mse" else calculate_bce_cost

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
        if len(ntk_gram_params) == 0 or i == len(df)-1 or np.linalg.norm(ntk_gram_params[-1] - params) >= MIN_NORM_CHANGE:
            ntk_gram = kernel_matrix(X, X, kernel=lambda x1, x2: ntk(x1, x2, params))
            ntk_grams.append(ntk_gram)
            ntk_gram_indexes.append(i)
            ntk_gram_params.append(params)

    return ntk_grams, ntk_gram_indexes


def calculate_pk(ntk_grams):
    return np.average(ntk_grams, axis=0)


def run_qnn(X, Y, loss, layers, epochs):

    N, D = X.shape
    print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Creating QNN ({layers} layers)")
    qnn = create_qnn(D, layers)
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Start training")
    specs, df = train_qnn(X, Y, qnn, loss, n_params=2*layers, epochs=epochs)
    specs["layers"] = layers
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Start NTK PK calculation")
    ntk_grams, ntk_gram_indexes = calculate_ntk(X, qnn, df)
    pk_gram = calculate_pk(ntk_grams)
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - End QNN")
    return specs, df, ntk_grams, ntk_gram_indexes, pk_gram


def run_qnns(D, snr, N, loss, MAX_LAYERS, MAX_EPOCHS):

    X, Y = create_gaussian_mixtures(D, snr, N)
    directory = f"experiment_snr{snr:0.2f}_d{D}_l{loss}_{datetime.now().strftime('%Y%m%d%H%M')}"
    Path(directory).mkdir(parents=True, exist_ok=True)

    for layers in range(1, MAX_LAYERS+1):
        specs, df, ntk_grams, ntk_gram_indexes, pk_gram = run_qnn(X, Y, loss, layers=layers, epochs=MAX_EPOCHS)
        specs["D"] = D
        specs["snr"] = snr
        specs["N"] = N
        specs["loss"] = loss
        specs["MAX_LAYERS"] = MAX_LAYERS
        specs["MAX_EPOCHS"] = MAX_EPOCHS
        json.dump(specs, open(f"{directory}/specs_{layers}.json", "w"))
        df.to_pickle(f"{directory}/trace_{layers}.pickle")
        np.save(f"{directory}/ntk_grams_{layers}.npy", ntk_grams)
        np.save(f"{directory}/ntk_gram_indexes_{layers}.npy", ntk_gram_indexes)
        np.save(f"{directory}/pk_gram_{layers}.npy", pk_gram)

# ========================================================================================
# ====================================== PLOTS ===========================================
# ========================================================================================


def center_kernel(K):
    K = K.copy()
    means = K.mean(axis=0)
    K -= means[None, :]
    K -= means[:, None]
    K += means.mean()
    return K


def calculate_tk_alignment(K1, K2, centered=False):
    if centered:
        K1 = center_kernel(K1)
        K2 = center_kernel(K1)
    return np.sum(K1 * K2) / np.linalg.norm(K1) / np.linalg.norm(K2)


def calculate_svc_accuracy(K, Y):
    regr = SVC(kernel='precomputed')
    regr.fit(K.T, Y)
    Y_actual = regr.predict(K.T)
    accuracy = np.sum(Y_actual == Y) / len(Y)
    return accuracy


def plot_dataset(X, Y):
    X1 = X[Y == 1]
    X2 = X[Y == -1]
    centroids = np.array([(.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5)])
    plt.scatter(X1[:, 0].tolist(), X1[:, 1].tolist(), label="First class", color='green')
    plt.scatter(X2[:, 0].tolist(), X2[:, 1].tolist(), label="Second class", color='blue')
    plt.scatter(centroids[:, 0].tolist(), centroids[:, 1].tolist(), label="Centroids", color='black', marker='x')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


def s2np(s):
    """String [v1 v2 ... vn] to NUMPY"""
    vs = s.replace("[", "").replace("]", "").split("\n")
    vf = [[float(f) for f in list(filter(lambda x: len(x) > 0, vrow.split(" ")))] for vrow in vs]
    npa = np.array(vf)
    if npa.shape[0] == 1:
        npa = npa.reshape(-1)
    return npa


def plot_model_training_loss_per_epoch(traces):
    """
    Plot the training loss of the many models
    X = epochs; Y = loss
    :param traces:
    :return:
    """
    MAX_DEPTH = len(traces)
    MAX_EPOCHS = len(traces[0])
    X = list(range(MAX_EPOCHS))
    color_palette = matplotlib.colormaps["autumn"](np.linspace(0, 1, MAX_DEPTH))
    plt.figure()
    for i, trace in enumerate(traces):
        Y = trace["loss"].to_numpy().astype('float')
        Y[np.isnan(Y)] = 0
        plt.plot(X, Y, color=color_palette[i], label=f"Depth {i+1}")
    plt.xlim((0, MAX_EPOCHS))
    plt.xlabel("Epochs of training")
    plt.ylabel("Loss")
    plt.legend()


def plot_model_params_norm_per_epoch(traces):
    """
    Plot the norm of params of the many models
    X = epochs; Y = loss
    :param traces:
    :return:
    """
    MAX_DEPTH = len(traces)
    MAX_EPOCHS = len(traces[0])
    color_palette = matplotlib.colormaps["autumn"](np.linspace(0, 1, MAX_DEPTH))
    plt.figure()
    for i, trace in enumerate(traces):
        init_params = trace["params"].loc[0]
        init_norm = np.linalg.norm(init_params)
        def normalise(x):
            return np.linalg.norm(x - init_params) / init_norm
        params_norm = np.vectorize(normalise)(trace["params"].to_numpy())
        plt.plot(range(MAX_EPOCHS), params_norm, color=color_palette[i], label=f"Depth {i+1}")
    plt.xlabel("Epochs of training")
    plt.ylabel(r"Norm change $\frac{||\theta(n)-\theta(0)||}{||\theta(0)||}$")
    plt.legend()
    plt.tight_layout()


def plot_tk_alignment_per_epoch(Y, ntk_grams_list, ntk_gram_indexes_list, pk_grams):
    """
    Plot the target kernel alignment per epoch
    X = epochs; Y = loss
    :param traces:
    :return:
    """
    M = len(Y)
    N = len(ntk_grams_list)
    YYt = Y.reshape((M,1)).dot(Y.reshape((1,M)))

    color_palette = matplotlib.colormaps["autumn"](np.linspace(0, 1, N))
    plt.figure()
    for i, (ntk_grams, ntk_indexes) in enumerate(zip(ntk_grams_list, ntk_gram_indexes_list)):
        x = ntk_indexes
        y = [calculate_tk_alignment(YYt, ntk_gram) for ntk_gram in ntk_grams]
        plt.plot(x, y, color=color_palette[i], label=f"NTK (depth {i + 1})")

    color_palette = matplotlib.colormaps["winter"](np.linspace(0, 1, N))
    for i in range(N):
        y = calculate_tk_alignment(YYt, pk_grams[i])
        plt.scatter([-100], [y], label=f"PK (depth {i+1})", color=color_palette[i])

    plt.xlabel("Epochs of training")
    plt.ylabel(r"Target-Kernel alignment")
    plt.legend()


def plot_accuracy_per_epoch(Y, ntk_grams_list, ntk_gram_indexes_list, pk_grams):
    """
    Plot the target kernel alignment per epoch
    X = epochs; Y = loss
    :param traces:
    :return:
    """
    M = len(Y)
    N = len(ntk_grams_list)
    YYt = Y.reshape((M,1)).dot(Y.reshape((1,M)))

    color_palette = matplotlib.colormaps["autumn"](np.linspace(0, 1, N))
    plt.figure()
    for i, (ntk_grams, ntk_indexes) in enumerate(zip(ntk_grams_list, ntk_gram_indexes_list)):
        x = ntk_indexes
        y = [calculate_svc_accuracy(ntk_gram, Y) for ntk_gram in ntk_grams]
        plt.plot(x, y, color=color_palette[i], label=f"NTK (depth {i + 1})")

    color_palette = matplotlib.colormaps["winter"](np.linspace(0, 1, N))
    for i in range(N):
        y = calculate_svc_accuracy(pk_grams[i], Y)
        plt.scatter([-100], [y], label=f"PK (depth {i+1})", color=color_palette[i])

    plt.xlabel("Epochs of training")
    plt.ylabel(r"Accuracy")
    plt.legend()


def plot_model_training_loss_per_depth(traces):
    """
    Plot the training loss of the many models
    X = epochs; Y = loss
    :param traces:
    :return:
    """
    MAX_DEPTH = len(traces)
    MAX_EPOCHS = len(traces[0])
    color_palette = matplotlib.colormaps["autumn"](np.linspace(0, 1, MAX_EPOCHS // 100 + 1))
    LOSSES_LIST = [traces[j]["loss"].to_numpy().astype('float') for j in range(MAX_DEPTH)]
    plt.figure()
    for i in range(0, MAX_EPOCHS+1, 100):
        X = np.array(list(range(1, MAX_DEPTH+1)))
        Y = np.array([LOSSES_LIST[j][i] for j in range(MAX_DEPTH)])
        Y[np.isnan(Y)] = 0.0
        plt.scatter(X, Y, color=color_palette[i // 100], label=f"Epoch {i}")
    plt.xticks(range(1, MAX_DEPTH+1))
    plt.xlabel("Depth")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()


def plot_model_parameter_norm_per_depth(traces):
    """
    Plot the training loss of the many models
    X = epochs; Y = loss
    :param traces:
    :return:
    """
    MAX_DEPTH = len(traces)
    MAX_EPOCHS = len(traces[0])
    color_palette = matplotlib.colormaps["autumn"](np.linspace(0, 1, MAX_EPOCHS // 100 + 1))
    get_param = lambda j, i: traces[j]["params"].loc[i]

    plt.figure()
    for i in range(0, MAX_EPOCHS+1, 100):
        params = [np.linalg.norm(get_param(j, i) - get_param(j, 0))/np.linalg.norm(get_param(j, 0)) for j in range(MAX_DEPTH)]
        plt.scatter(range(1, MAX_DEPTH+1), params, color=color_palette[i // 100], label=f"Epoch {i}")
    plt.xticks(range(1, MAX_DEPTH+1))
    plt.xlabel("Depth")
    plt.ylabel(r"Norm change $\frac{||\theta(n)-\theta(0)||}{||\theta(0)||}$")
    plt.legend()
    plt.tight_layout()


def plot_tk_alignment_per_depth(Y, ntk_grams_list, ntk_gram_indexes_list, pk_grams):
    """
    Plot the target kernel alignment per epoch
    X = epochs; Y = loss
    :param traces:
    :return:
    """
    M = len(Y)
    N = len(ntk_grams_list)
    YYt = Y.reshape((M,1)).dot(Y.reshape((1,M)))
    plt.figure()

    color_palette = matplotlib.colormaps["autumn"](np.linspace(0, 1, N))
    for i in range(N):
        y = calculate_tk_alignment(YYt, ntk_grams_list[i][-1])
        plt.scatter([i+1], [y], label=f"NTK (depth {i+1})", color=color_palette[i])

    color_palette = matplotlib.colormaps["winter"](np.linspace(0, 1, N))
    for i in range(N):
        y = calculate_tk_alignment(YYt, pk_grams[i])
        plt.scatter([i+1], [y], label=f"PK (depth {i+1})", color=color_palette[i])

    plt.xlabel("Depth")
    plt.ylabel(r"Target-Kernel alignment")
    plt.legend()


def plot_accuracy_per_depth(Y, ntk_grams_list, ntk_gram_indexes_list, pk_grams):
    """
    Plot the target kernel alignment per epoch
    X = epochs; Y = loss
    :param traces:
    :return:
    """
    M = len(Y)
    N = len(ntk_grams_list)
    YYt = Y.reshape((M,1)).dot(Y.reshape((1,M)))
    plt.figure(figsize=(5, 5))

    color_palette = matplotlib.colormaps["autumn"](np.linspace(0, 1, N))
    for i in range(N):
        y = calculate_svc_accuracy(ntk_grams_list[i][-1], Y)
        if y < 0:
            print(f"NTK gram matrix {i} has y={y}<0")
        plt.scatter([i+1], [y], label=f"NTK (depth {i+1})", color=color_palette[i])

    color_palette = matplotlib.colormaps["winter"](np.linspace(0, 1, N))
    for i in range(N):
        y = calculate_svc_accuracy(pk_grams[i], Y)
        plt.scatter([i+1], [y], label=f"PK (depth {i+1})", color=color_palette[i])

    plt.xlabel("Depth")
    plt.ylabel(r"Accuracy")
    plt.legend(bbox_to_anchor=(1, 1), prop={'size': 6})
    plt.tight_layout()

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
@click.option('--loss', type=click.Choice(['mse', 'bce']), required=True)
@click.option('--layers', default=20, type=int)
@click.option('--epochs', default=1000, type=int)
def experiment(d, snr, n, loss, layers, epochs):
    """
    Start the experiments
    :param d: dimensionality of the data (at least 2
    :param snr: signal to noise ratio
    :param n: number of training samples (must be multiple of 4, suggested and default 16)
    :param loss: MSE (mean square error) or BCE (binary cross entropy)
    :param layers: maximum number of layers (default 20)
    :param epochs: maximum number of training epochs (default 1000)
    :return: nothing, everything is saved to file
    """
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Experiment D={d}, snr={snr}, N={n}, loss={loss}, MAX_LAYERS={layers}, MAX_EPOCHS={epochs}")
    run_qnns(d, snr, n, loss, MAX_LAYERS=layers, MAX_EPOCHS=epochs)


@main.command()
@click.option('--directory', type=click.Path(exists=True))
def analyze(directory):
    """
    Analyze the data contained in the given directory
    :param directory: where the experiment data is saved
    :return: nothing, everything is saved to file
    """
    # create analysis directory and load specifications
    subdirectory = directory + "/analysis"
    Path(subdirectory).mkdir(parents=True, exist_ok=True)
    specs = json.load(open(f"{directory}/specs_1.json"))
    X, Y, D, snr, N, loss = s2np(specs["X"]), s2np(specs["Y"]), int(specs["D"]), float(specs["snr"]), int(specs["N"]), specs["loss"]

    # load trace data
    layers_files = list(
        filter(lambda x: x.startswith("trace"), [x.name for x in Path(directory).iterdir() if x.is_file()]))
    MAX_LAYERS = len(layers_files)
    TRACES = [pd.read_pickle(f"{directory}/trace_{l}.pickle") for l in range(1, MAX_LAYERS + 1)]
    MAX_DEPTH = len(TRACES[0])

    # load gram matrices
    ntk_grams_list = [np.load(f"{directory}/ntk_grams_{l}.npy") for l in range(1, MAX_LAYERS + 1)]
    ntk_gram_indexes_list = [np.load(f"{directory}/ntk_gram_indexes_{l}.npy") for l in range(1, MAX_LAYERS + 1)]
    pk_gram_list = [np.load(f"{directory}/pk_gram_{l}.npy") for l in range(1, MAX_LAYERS + 1)]

    # plot dataset
    plot_dataset(X, Y)
    plt.title("Gaussian Mixtures dataset")
    dataset_info = f"Dimensionality D={D}, signal noise ratio snr={snr}, size N={N}"
    plt.figtext(0.5, 0, dataset_info, wrap=True, horizontalalignment='center', verticalalignment='bottom', fontsize=12)
    plt.savefig(f"{subdirectory}/dataset_plot.png", dpi=300, format='png')

    # loss of the models at the various depths (last epochs)
    plot_model_training_loss_per_epoch(TRACES)
    plt.title(f"Loss (training set) of variational models (loss={loss})")
    plt.savefig(f"{subdirectory}/loss_in_training_per_epoch.png", dpi=300, format='png')

    # loss of each model during the training (one single plot)
    plot_model_training_loss_per_depth(TRACES)
    plt.title(f"Loss (training set) of variational models (loss={loss})")
    plt.savefig(f"{subdirectory}/loss_in_training_per_depth.png", dpi=300, format='png')

    # (end - start) norm change of the models at the various depths (all lines in one plot, x=epoch, y=norm change)
    plot_model_params_norm_per_epoch(TRACES)
    plt.title(f"Norm change during training of variational models (loss={loss})")
    plt.savefig(f"{subdirectory}/param_norm_change_in_training_per_epoch.png", dpi=300, format='png')

    # norm change of each parameter, of each model
    plot_model_parameter_norm_per_depth(TRACES)
    plt.title(f"Norm change during training of variational models (loss={loss})")
    plt.savefig(f"{subdirectory}/param_norm_change_in_training_per_depth.png", dpi=300, format='png')

    # # target-kernel alignment of each NTK during the training + PK
    # plot_tk_alignment_per_epoch(Y, ntk_grams_list, ntk_gram_indexes_list, pk_gram_list)
    # plt.title(f"Target-kernel alignment during training of NTK and PK (loss={loss})")
    # plt.savefig(f"{subdirectory}/target_kernel_alignment_in_training_per_epoch.png", dpi=300, format='png')

    # # target-kernel alignment of the last epoch NTK vs PK (varying the depth)
    # plot_tk_alignment_per_depth(Y, ntk_grams_list, ntk_gram_indexes_list, pk_gram_list)
    # plt.title(f"Target-kernel alignment during training of NTK and PK (loss={loss})")
    # plt.savefig(f"{subdirectory}/target_kernel_alignment_in_training_per_depth.png", dpi=300, format='png')

    # SVM model accuracy of each NTK during the training + PK
    plot_accuracy_per_epoch(Y, ntk_grams_list, ntk_gram_indexes_list, pk_gram_list)
    plt.title(f"SVM accuracy during training of NTK and PK (loss={loss})")
    plt.savefig(f"{subdirectory}/accuracy_in_training_per_epoch.png", dpi=300, format='png')

    # SVM model accuracy of the last epoch NTK vs PK (varying the depth)
    plot_accuracy_per_depth(Y, ntk_grams_list, ntk_gram_indexes_list, pk_gram_list)
    plt.title(f"SVM accuracy during training of NTK and PK (loss={loss})")
    plt.savefig(f"{subdirectory}/accuracy_in_training_per_depth.png", dpi=300, format='png')


@main.command()
def report():
    """
    Generate report in html format
    :return: nothing, the html il saved to report_<datetime>.html
    """
    regex = re.compile(r"experiment_snr([0-9.]*)_d([0-9]*)_l([a-z]*)_[0-9]*")
    experiments_list = [x.name for x in Path(".").iterdir() if x.is_dir() and x.name.startswith("experiment_")]
    experiments_specs = [(regex.match(experiment), experiment) for experiment in experiments_list]
    experiments_specs = [{'snr': r.group(1), 'd': r.group(2), 'loss': r.group(3), 'dir': dir} for (r, dir) in experiments_specs]
    filtered_specs = lambda key, value: [spec for spec in experiments_specs if spec[key] == value]
    title = "Gaussian Mixtures with Quantum Machine Learning models and Path Kernel"
    gen_time = datetime.now()
    rprt = f"""
<html>
    <head>
        <title>{title}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
        <style>img{{max-width: 600px}}</style>
    </head>
    <body class="container">
        <h1>{title}</h1>
        <p>Generated at {gen_time.strftime('%d/%m/%Y %H:%M:%S')}</p>
        <h2>Dataset</h2>
        <p>The dataset is composed of N samples. Each sample has D components in the form (x1, x2, 0, 0, ..., 0)
        where x1 and x2 are the coordinated of one of the four centroids (+-.5, +-.5) plus some noise that I've called
        snr (signal to noise) but I'm not sure it matches the definition... Is just noise independently sampled from
        a gaussian having mean zero and variance equal to the 'snr'. The higher the 'snr', the more noisy my dataset
        is and the more difficult is to classify the samples. 
        In Refinetti's work, having large value to D results in a difficult environment for random feature kernel models
        while Neural Networks, due to their dissipative work, can easily learn the distribution reaching the optimal
        performance of the oracle. </p>
        {"".join(f"<p>Dataset generated with snr={spec['snr']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/dataset_plot.png'/></p>{chr(10)}" 
                 for spec in experiments_specs)}
        <br/>
        <h2>Loss</h2>
        <p>The loss is the function minimized during the training phase by the optimizer. In classical deep learning 
        theory, I should reach zero loss when I have just enough parameters to perfectly fit the data, resulting in 
        a large generalization error. Due to double descent and the implic regularization of gradient descent 
        optimization, adding further parameters still find a solution having zero loss which although represent a 
        simpler function, with better generalization performances.<p>
        <p>The loss function we can study is either the Binary Cross Entropy, which is used for classification 
        problems such this one, and the Mean Square Error, which is used in regression problems usually but it still
        make sense to make the comparison.</p>
        <h4>Loss per epoch</h4>
        <p>The following paragram study the loss of each model with respect to the epoch of training (x axis). The
        color of the line represents how many layer the QNN has.</p>
        <h6>Using loss BCE (Binary Cross Entropy)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/loss_in_training_per_epoch.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'bce'))}
        <h6>Using loss MSE (Mean Square Error)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/loss_in_training_per_epoch.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'mse'))}
        <br/>
        <h4>Loss per depth</h4>
        <p>The following paragram study the loss of each model with respect to the depth (x axis). The point on each 
        vertical line represents the evolution of the loss at a certain depth, during the training each epoch multiple
        of 100. The fact that the points are not evenly spread from the top to the bottom means that the loss 
        immediately reach zero (well, in 100 epoch at least).</p>
        <h6>Using loss BCE (Binary Cross Entropy)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/loss_in_training_per_depth.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'bce'))}
        <h6>Using loss MSE (Mean Square Error)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/loss_in_training_per_depth.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'mse'))}
        <br/>
        <h2>Parameters norm change</h2>
        <p>Parameters norm change highlight the lazy training phenomena: if the parameters stay close to their 
        initialization then we are in a lazy training regime (a sort of, since our QNN are indeed linear model, 
        can be really call them feature learning vs lazy regimes? isn't that just optimization?).
        <h4>Parameters norm change per epoch</h4>
        <p>These plots highlight the fact that after a (big or small) adjustment of the parameters, they converges 
        quickly into a solution (n.b. note that here just the norm is checked, eventually if they are rotating they
        will indeed change if I can't see that from this plot... It's more a technical note though). 
        <h6>Using loss BCE (Binary Cross Entropy)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/param_norm_change_in_training_per_epoch.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'bce'))}
        <h6>Using loss MSE (Mean Square Error)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/param_norm_change_in_training_per_epoch.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'mse'))}
        <br/>
        <h4>Parameters norm change per depth</h4>
        <p>If the yellow-er dots stay close to the red (initial) ones then we are in lazy training. It almost never
        happen than the norm first increase (going far from the initialization) then decreases (returning back to the
        initialization).
        <h6>Using loss BCE (Binary Cross Entropy)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/param_norm_change_in_training_per_depth.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'bce'))}
        <h6>Using loss MSE (Mean Square Error)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/param_norm_change_in_training_per_depth.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'mse'))}
        <br/>
        <h2>Accuracy</h2>
        <p>We compare the accuracy of the model in predicting the labels by comparing the NTK calculated (for each QNN)
        at its last configuration, at the end of the training (1000 epoch) and the Path Kernel. 
        The red-to-yellow line and dots are related to the Neural Tangent Kernel (1000 epoch) + SVM model.
        The blue-to-green line and dots are related to the Path Kernel + SVM model. </p>
        <p>Accuracy is calculated by still using the same dataset of the training (i.e. if I use the non-linearized
        variational model instead of quantum kernel+SVM I will get zero error due to the zero loss). Considering a 
        testing set after this preliminary results is mandatory. 
        <h4>Accuracy per epoch</h4>
        <p>We surely expect the accuracy is higher for small snr. <b>Preliminary thoughts</b>: it seems that with
        small depths the NTK performs better, and with higher depths the NTK performs worse. I was expecting the
        opposite, since the PK at small depth allows to interpret the model as a kernel machine while the NTK at small
        depth (and thus larger parameters norm change) means nothing. We need to check it here something. Well, to tell
        the truth, it seems to me that the performances of PK are almost the average of the performance of NTK...<br/>
        P.S: this graph are not great. Do we know a fancier graphical representation?</p>
        <h6>Using loss BCE (Binary Cross Entropy)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/accuracy_in_training_per_epoch.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'bce'))}
        <h6>Using loss MSE (Mean Square Error)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/accuracy_in_training_per_epoch.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'mse'))}
        <br/>
        <h4>Accuracy per depth</h4>
        <p>Much clearer representation... From these graphs, PK seems to win in general, especially with BCE loss! 
        Still, we are using the same points of the training.</p>
        <h6>Using loss BCE (Binary Cross Entropy)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/accuracy_in_training_per_depth.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'bce'))}
        <h6>Using loss MSE (Mean Square Error)</h6>
        {"".join(f"<p>Experiment having snr={spec['snr']}, d={spec['d']}, loss={spec['loss']}:<br/>"
                 f"<img src='{spec['dir']}/analysis/accuracy_in_training_per_depth.png'/></p>{chr(10)}" 
                 for spec in filtered_specs('loss', 'mse'))}
        <br/>
        <h2>By increasing D...</h2>
        <p>Does it happens that, like in Refinetti's work, by increasing D the kernel machine loses performances? 
        If that happens, it might mean that the kernel are working just like random features. We still miss some further
        experiments to see a pattern, however:</p>
        <p>Take loss=MSE, snr=0.20, D=5..6</p>
        <p>TODO</p>
        <br/><br/><br/><p>End of report</p><br/><br/><br/>
    </body>
</html>
    """
    print(rprt, file=open(f"report_{gen_time.strftime('%Y%m%d%H%M')}.html", "w"))


if __name__ == '__main__':
    main()

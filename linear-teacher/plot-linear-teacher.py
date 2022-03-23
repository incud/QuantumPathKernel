import matplotlib.colors
import numpy as np
import pandas.core.frame
import pennylane as qml
from pennylane import numpy as pnp  # -> substituted by JAX
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import json
import click

from sklearn.svm import SVR



training_losses = []
testing_losses = []





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


def create_params_norm_change(dfs):
    EPOCHS = len(dfs[0].params)
    params_change = pnp.zeros((len(dfs), EPOCHS))
    for i, df in enumerate(dfs):
        P = df.loc[0].params
        NP = np.linalg.norm(P)
        def normalise(x):
            return np.linalg.norm(x - P) / NP
        df_norm = np.vectorize(normalise)(df.params.to_numpy())
        params_change[i] = df_norm
    params_change[params_change < 1e-2] = 1e-2
    return params_change


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



def params_norm_change_log(dfs):

    if type(dfs[0]) == pandas.core.frame.DataFrame:
        params_hm = create_params_norm_change(dfs)
    else:
        N = len(dfs)
        params_hm = create_params_norm_change(dfs[0])
        for i in range(1, N):
            params_hm += create_params_norm_change(dfs[i])
        params_hm /= N

    im = plt.pcolor(params_hm.T, norm=matplotlib.colors.LogNorm(vmin=1e-2))
    plt.colorbar(im, orientation='horizontal')
    plt.yscale('log')
    plt.ylim((1, 1000))
    plt.axvline(x=4, color='r')
    plt.show()

# ======================================================================================================

TRAINING_SET_SIZE = 6
TESTING_SET_SIZE = 20
MAX_REP = 15
MAX_DEPTH = 60
LINEAR_W = 0.66
MAX_EPOCHS = 1001
DATAFRAMES = []
DATAFRAMES.append([])  # the zero-th repetition is empty


def load_trace(rep, layers):
    return pd.read_pickle(f"linear-teacher-experiments-{rep}/trace_{layers}.pickle")


def load_all_traces():
    print("Loading dataframes...")
    for rep in range(1, MAX_REP + 1):
        DATAFRAMES.append([])
        for layers in range(1, MAX_DEPTH + 1):
            df = load_trace(rep=rep, layers=layers)
            DATAFRAMES[rep].append(df)
    print("Loaded!")

@click.group()
def main():
    """Simple CLI for querying books on Google Books by Oyetoke Toby"""
    pass


@main.command()
@click.option('--repetition', default=-1, type=int)
def plotnormchangeheatmap(repetition):
    """
    Plot the heatmap related in norm change

    REPETITION is which repetition you want the plot of (-1 for the average)
    """
    load_all_traces()

    def create_norm_change_heatmap(df_nested_list):
        # create empty heatmap
        params_change = pnp.zeros((MAX_DEPTH, MAX_EPOCHS))
        # for each "depth" dataframe
        for i, df in enumerate(df_nested_list):
            init_params = df.loc[0].params
            init_params_norm = np.linalg.norm(init_params)

            def normalise(actual_params):
                return np.linalg.norm(actual_params - init_params) / init_params_norm

            df_norm = np.vectorize(normalise)(df.params.to_numpy())
            params_change[i] = df_norm
        params_change[params_change < 1e-2] = 1e-2
        return params_change

    if repetition == -1:  # average of all repetitions
        heatmap_matrix = np.average([create_norm_change_heatmap(DATAFRAMES[i]) for i in range(MAX_REP)], axis=0)
    else:  # a single, specific repetition
        heatmap_matrix = create_norm_change_heatmap(DATAFRAMES[repetition])

    im = plt.pcolor(heatmap_matrix.T, norm=matplotlib.colors.LogNorm(vmin=1e-2))
    plt.title(f"Norm change ({'Average of all repetitions' if repetition==-1 else f'repetition {repetition}'})")
    plt.colorbar(im, orientation='horizontal')
    plt.ylabel('Epochs of training')
    plt.yscale('log')
    plt.ylim((1, 1000))
    plt.xlabel('Depth of circuits')
    plt.axvline(x=4, color='r')
    plt.show()


@main.command()
@click.option('--repetition', default=-1, type=int)
@click.option('--epoch', default=0, type=int)
@click.option('--depth', default=0, type=int)
def plotnormchange(repetition, epoch, depth):
    """
    Plot the norm change

    REPETITION is which repetition you want the plot of (-1 for the average)
    """
    if epoch == depth:
        raise ValueError("You must set either EPOCH or DEPTH")
    raise ValueError("TODO")
    load_all_traces()


def s2np(s):
    return np.array(s[1:-1].split(), dtype=float)


def target_kernel_alignment(y1, y2, K):
    """Calculate kernel polarity"""
    y1 = y1.reshape((len(y1), 1))
    y2 = y2.reshape((1, len(y2)))
    Y = np.matmul(y1, y2)
    polarity = np.sum(Y * K)
    norm_Y = np.linalg.norm(Y)
    norm_K = np.linalg.norm(K)
    target_alignment = polarity / (norm_K * norm_Y)
    return target_alignment


def svr_loss(K_train, Y_train, K_test, Y_test):
    regr = SVR(kernel='precomputed')
    regr.fit(K_train.T, Y_train)
    Y_actual = regr.predict(K_test.T)
    return np.linalg.norm(Y_actual - Y_test)


def load_ntk_data(repetition, layers, epoch):
    specs = json.load(open(f"linear-teacher-experiments-{repetition}/specs_{layers}.json"))
    X_train, X_test = s2np(specs["X_train"]), s2np(specs["X_test"])
    Y_train, Y_test = s2np(specs["Y_train"]), s2np(specs["Y_test"])
    ntk_training_grams = np.load(f"linear-teacher-experiments-{repetition}/ntk_training_gram_{layers}.npy")
    ntk_testing_grams = np.load(f"linear-teacher-experiments-{repetition}/ntk_testing_gram_{layers}.npy")
    if epoch == -1:
        K_train = np.average(ntk_training_grams, axis=2)
        # print(ntk_training_grams.shape, "=>", K_train.shape)
        K_test = np.average(ntk_testing_grams, axis=2)
        # print(ntk_testing_grams.shape, "=>", K_test.shape)
    else:
        K_train = ntk_training_grams[:, :, epoch]
        K_test = ntk_testing_grams[:, :, epoch]
    return X_train, K_train, Y_train, X_test, K_test, Y_test


@main.command()
@click.option('--repetition', default=-1, type=int)
@click.option('--epoch', default=0, type=int)
def plotntktargetalignment(repetition, epoch):
    """
    Plot the norm change

    REPETITION is which repetition you want the plot of (-1 for the average)
    """
    if repetition != 1:
        raise ValueError("Not simulated yet")

    MAX_DEPTH_NTK = 28

    train_alignment = []
    test_alignment = []

    for layers in range(1, MAX_DEPTH_NTK+1):
        _, K_train, Y_train, _, K_test, Y_test = load_ntk_data(repetition, layers, epoch)
        train_alignment.append(target_kernel_alignment(Y_train, Y_train, K_train))
        test_alignment.append(target_kernel_alignment(Y_train, Y_test, K_test))

    plt.title(f"Target-Kernel alignment using NTK at epoch {epoch}")
    plt.plot(range(1, MAX_DEPTH_NTK+1), train_alignment, label="Training")
    plt.plot(range(1, MAX_DEPTH_NTK+1), test_alignment, label="Testing")
    plt.legend()
    plt.show()


@main.command()
@click.option('--repetition', default=-1, type=int)
@click.option('--epoch', default=0, type=int)
def plotntksvrloss(repetition, epoch):
    """
    Plot the norm change

    REPETITION is which repetition you want the plot of (-1 for the average)
    """
    if repetition != 1:
        raise ValueError("Not simulated yet")

    MAX_DEPTH_NTK = 28

    accuracies = []

    for layers in range(1, MAX_DEPTH_NTK+1):
        _, K_train, Y_train, _, K_test, Y_test = load_ntk_data(repetition, layers, epoch)
        accuracies.append(svr_loss(K_train, Y_train, K_test, Y_test))

    plt.title(f"SVR loss using NTK at epoch {epoch} - lower is better")
    plt.plot(range(1, MAX_DEPTH_NTK+1), accuracies, label="Loss")
    plt.legend()
    plt.show()


@main.command()
@click.option('--repetition', default=-1, type=int)
@click.option('--epoch', default=0, type=int)
@click.option('--layers', default=1, type=int)
def plotntkprediction(repetition, epoch, layers):
    """
    Plot the norm change

    REPETITION is which repetition you want the plot of (-1 for the average)
    """
    MAX_DEPTH_NTK = 28

    if repetition != 1:
        raise ValueError("Not simulated yet")
    if layers not in range(1, MAX_DEPTH_NTK+1):
        raise ValueError("Not simulated yet")

    X_train, K_train, Y_train, X_test, K_test, Y_test = load_ntk_data(repetition, layers, epoch)
    regr = SVR(kernel='precomputed')
    regr.fit(K_train.T, Y_train)
    Y_actual = regr.predict(K_test.T)

    plt.title(f"SVR loss using NTK at epoch {epoch} - lower is better")
    print(X_train.tolist())
    print(Y_train.tolist())
    print(X_test.tolist())
    print(Y_test.tolist())
    plt.plot([0,1], [0,0.66], color='red')
    plt.scatter(X_train.tolist(), Y_train.tolist(), label="Training")
    plt.scatter(X_test.tolist(), Y_test.tolist(), label="Testing (ideal)")
    plt.scatter(X_test.tolist(), Y_actual.tolist(), label="Testing (actual)")
    plt.legend()
    plt.show()


@main.command()
def fixspecs():
    """
    Plot the norm change

    REPETITION is which repetition you want the plot of (-1 for the average)
    """
    pass


if __name__ == '__main__':
    main()

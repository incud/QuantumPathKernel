import matplotlib.pyplot as plt
import numpy as np
import optax
from gaussian_mixtures import create_gaussian_mixtures_with_garbage
from mlp_classifier import RandomKernelClassifier, NeuralNetworkClassifier, DeepNonLinearClassifier
from plot_decision_boundaries import plot_safe_decision_areas, plot_safe_decision_boundaries, make_meshgrid

np.random.seed(9182736)
DIR = "gaussian_mixtures_test_2_images"
N = 100
EPOCHS = 10_000
optimizer = optax.adam(1e-3)


def plot(ax, clf, clf_name, X, Y, D, noise):
    grid_x, grid_y = make_meshgrid(np.array([-1, 1]), np.array([-1, 1]), h=0.01)
    n_elements = grid_x.ravel().shape[0]
    the_x = np.random.normal(loc=0.0, scale=noise, size=(n_elements, D))
    the_x[:, 0] = grid_x.ravel()
    the_x[:, 1] = grid_y.ravel()
    grid_z = clf.predict(the_x).reshape(grid_x.shape)
    title = f"Decision boundaries - D={D} - noise={noise} - {clf_name} (#params {clf.get_num_parameters()})"
    plot_safe_decision_areas(ax, X, Y, grid_x, grid_y, grid_z, title=title)


for D in [2, 10, 50]:

    for noise in [0.1, 0.5, 1.0]:

        DIR2 = f"{DIR}/D{D}_noise{noise}_N{N}"
        X, Y = create_gaussian_mixtures_with_garbage(D, noise, N)
        Y[Y == -1] = 0
        np.save(f"{DIR2}_X.npy", X)
        np.save(f"{DIR2}_Y.npy", Y)

        fig, axs = plt.subplots(2, 4)

        nn_model_1 = DeepNonLinearClassifier([5, 3], 2, optimizer, EPOCHS, seed=12345)
        nn_model_1.fit(X, Y)
        plot(axs[0, 0], nn_model_1, "NN(5 3)", X, Y, D, noise)

        nn_model_2 = NeuralNetworkClassifier(3, 2, optimizer, EPOCHS, seed=12345)
        nn_model_2.fit(X, Y)
        plot(axs[0, 1], nn_model_2, "NN(3)", X, Y, D, noise)

        nn_model_3 = NeuralNetworkClassifier(5, 2, optimizer, EPOCHS, seed=12345)
        nn_model_3.fit(X, Y)
        plot(axs[0, 2], nn_model_3, "NN(5)", X, Y, D, noise)

        nn_model_4 = NeuralNetworkClassifier(10, 2, optimizer, EPOCHS, seed=12345)
        nn_model_4.fit(X, Y)
        plot(axs[0, 3], nn_model_4, "NN(10)", X, Y, D, noise)

        lin_model_1 = RandomKernelClassifier(3, 2, optimizer, EPOCHS, seed=12345)
        lin_model_1.fit(X, Y)
        plot(axs[1, 0], lin_model_1, "RK(3)", X, Y, D, noise)

        lin_model_2 = RandomKernelClassifier(5, 2, optimizer, EPOCHS, seed=12345)
        lin_model_2.fit(X, Y)
        plot(axs[1, 1], lin_model_2, "RK(5)", X, Y, D, noise)

        lin_model_3 = RandomKernelClassifier(10, 2, optimizer, EPOCHS, seed=12345)
        lin_model_3.fit(X, Y)
        plot(axs[1, 2], lin_model_3, "RK(10)", X, Y, D, noise)

        lin_model_4 = RandomKernelClassifier(100, 2, optimizer, EPOCHS, seed=12345)
        lin_model_4.fit(X, Y)
        plot(axs[1, 3], lin_model_4, "RK(100)", X, Y, D, noise)

        fig.savefig(f"{DIR2}.png")
        plt.close(fig)

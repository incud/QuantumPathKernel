import numpy as np
import optax
from gaussian_mixtures import create_gaussian_mixtures_with_garbage
from mlp_classifier import DeepLinearClassifier, DeepNonLinearClassifier
from plot_decision_boundaries import plot_safe_decision_areas, plot_safe_decision_boundaries, make_meshgrid

np.random.seed(9182736)
DIR = "gaussian_mixtures_test_1_images"
N = 100
EPOCHS = 10_000


def plot(clf, clf_name, X, Y, D, noise):
    grid_x, grid_y = make_meshgrid(np.array([-1, 1]), np.array([-1, 1]), h=0.01)
    n_elements = grid_x.ravel().shape[0]
    the_x = np.random.normal(loc=0.0, scale=noise, size=(n_elements, D))
    the_x[:, 0] = grid_x.ravel()
    the_x[:, 1] = grid_y.ravel()
    grid_z = clf.predict(the_x).reshape(grid_x.shape)
    print(np.sum(grid_z == 1), np.sum(grid_z == -1))
    title = f"Decision boundaries - D={D} - noise={noise} - {clf_name} (#params {clf.get_num_parameters()})"
    fig = plot_safe_decision_boundaries(X, Y, grid_x, grid_y, grid_z, title=title)
    fig.savefig(f"{DIR2}_{clf_name}_dec_bounds.png")
    plot_safe_decision_areas(X, Y, grid_x, grid_y, grid_z, title=title)
    fig.savefig(f"{DIR2}_{clf_name}_dec_area.png")


for D in [2, 10, 50]:

    for noise in [0.1, 0.5, 1.0]:

        DIR2 = f"{DIR}/D{D}_noise{noise}_N{N}"
        X, Y = create_gaussian_mixtures_with_garbage(D, noise, N)
        Y[Y == -1] = 0
        np.save(f"{DIR2}_X.npy", X)
        np.save(f"{DIR2}_Y.npy", Y)
        optimizer = optax.adam(1e-3)

        nn_model_1 = DeepNonLinearClassifier([3], 2, optimizer, EPOCHS, seed=12345)
        nn_model_1.fit(X, Y)
        plot(nn_model_1, "NN(3 2)", X, Y, D, noise)

        nn_model_2 = DeepNonLinearClassifier([3, 3], 2, optimizer, EPOCHS, seed=12345)
        nn_model_2.fit(X, Y)
        plot(nn_model_2, "NN(3 3 2)", X, Y, D, noise)

        nn_model_3 = DeepNonLinearClassifier([5], 2, optimizer, EPOCHS, seed=12345)
        nn_model_3.fit(X, Y)
        plot(nn_model_3, "NN(5 2)", X, Y, D, noise)

        nn_model_4 = DeepNonLinearClassifier([5, 2], 2, optimizer, EPOCHS, seed=12345)
        nn_model_4.fit(X, Y)
        plot(nn_model_4, "NN(5 5 2)", X, Y, D, noise)

        lin_model_1 = DeepLinearClassifier([3], 2, optimizer, EPOCHS, seed=12345)
        lin_model_1.fit(X, Y)
        plot(lin_model_1, "LIN(3 2)", X, Y, D, noise)

        lin_model_2 = DeepLinearClassifier([3, 3], 2, optimizer, EPOCHS, seed=12345)
        lin_model_2.fit(X, Y)
        plot(lin_model_2, "LIN(3 3 2)", X, Y, D, noise)

        lin_model_3 = DeepLinearClassifier([10], 2, optimizer, EPOCHS, seed=12345)
        lin_model_3.fit(X, Y)
        plot(lin_model_3, "LIN(10 2)", X, Y, D, noise)

        lin_model_4 = DeepLinearClassifier([100], 2, optimizer, EPOCHS, seed=12345)
        lin_model_4.fit(X, Y)
        plot(lin_model_4, "LIN(100 2)", X, Y, D, noise)

        # lin_model_5 = DeepLinearClassifier([100], 2, optimizer, EPOCHS, seed=12345)
        # lin_model_5.fit(X, Y)
        # plot(lin_model_5, "LIN(100 2)", X, Y, D, noise)
#
        # lin_model_6 = DeepLinearClassifier([200], 2, optimizer, EPOCHS, seed=12345)
        # lin_model_6.fit(X, Y)
        # plot(lin_model_6, "LIN(200 2)", X, Y, D, noise)
#
        # lin_model_7 = DeepNonLinearClassifier([500], 2, optimizer, EPOCHS, seed=12345)
        # lin_model_7.fit(X, Y)
        # plot(lin_model_7, "LIN(500 2)", X, Y, D, noise)

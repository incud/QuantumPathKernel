import numpy as np


def create_gaussian_mixtures(D, noise, N):
    """
    Create the Gaussian mixture dataset
    :param D: number of dimensions: (x1, x2, 0, .., 0) in R^D
    :param noise: intensity of the random noise (mean 0)
    :param N: number of elements to generate
    :return: dataset
    """
    if N % 4 != 0:
        raise ValueError("The number of elements within the dataset must be a multiple of 4")
    if D < 2:
        raise ValueError("The number of dimensions must be at least 2")
    if noise < 0:
        raise ValueError("Signal to noise ratio must be > 0")

    X = np.zeros((N, D))
    Y = np.zeros((N,))
    centroids = np.array([(.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5)])
    for i in range(N):
        quadrant = i % 4
        Y[i] = 1 if quadrant % 2 == 0 else -1  # labels are 0 or 1
        X[i][0], X[i][1] = centroids[quadrant] + np.random.uniform(-noise, noise, size=(2,))
    return X, Y


def create_gaussian_mixtures_with_garbage(D, noise, N):
    """
    Create the Gaussian mixture dataset with noise
    :param D: number of dimensions: (x1, x2, 0, .., 0) in R^D
    :param noise: intensity of the random noise (mean 0)
    :param N: number of elements to generate
    :return: dataset
    """
    X, Y = create_gaussian_mixtures(D, noise, N)
    X[:, 2:] = np.random.normal(scale=noise, size=X[:, 2:].shape)
    return X, Y

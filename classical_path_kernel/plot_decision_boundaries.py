import matplotlib.pyplot as plt
import numpy as np


def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_decision_boundaries(clf, X, y, save_path, precision=.1):
    plt.close('all')
    plt.clf()
    fig, ax = plt.subplots()
    grid_x, grid_y = make_meshgrid(X[:, 0], X[:, 1], h=precision)
    Z = clf.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    Z = Z.reshape(grid_x.shape)
    np.save(f"{save_path}_Z.npy", Z)
    out = ax.contourf(grid_x, grid_y, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('y label here')
    ax.set_xlabel('x label here')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Decision boundaries")
    fig.savefig(f"{save_path}.png")
    plt.close('all')
    plt.clf()


def plot_decision_areas(clf, X, y, load_path, save_path):
    plt.close('all')
    plt.clf()
    fig, ax = plt.subplots()
    grid_x, grid_y = make_meshgrid(X[:, 0], X[:, 1])
    if load_path is None:
        Z = clf.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    else:
        Z = np.load(load_path + "_Z.npy")
    Z = Z.reshape(grid_x.shape)
    out = ax.scatter(grid_x, grid_y, c=Z, cmap=plt.cm.coolwarm, alpha=0.3, s=400)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('y label here')
    ax.set_xlabel('x label here')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Decision boundaries")
    fig.savefig(f"{save_path}.png")
    plt.close('all')
    plt.clf()


# from sklearn.svm import SVC
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets
#
# iris = datasets.load_iris()
# # Select 2 features / variable for the 2D plot that we are going to create.
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target
# model = svm.SVC(kernel='linear')
# clf = model.fit(X, y)
# plot_decision_boundaries(clf, X, y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KernelCenterer, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import svm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, polynomial_kernel


def kernel_alignment(K1, K2):
    """Calculate kernel alignment between K1 and K2
    Return: perfect alignment is +1; alignment 0 is no alignment
    """
    return np.sum(K1 * K2) / np.linalg.norm(K1) / np.linalg.norm(K2)


def kernel_target_alignment(K, y):
    """Calculate kernel-target alignment between K and y*y^T, where y_i = +1/-1
    Return: perfect alignment has magnitude 1 (both +1 or -1 are ok); alignment 0 is no alignment
    """
    y = y.reshape((len(y), 1))
    yyt = np.matmul(y, y.T)
    return kernel_alignment(K, yyt)


def build_gram_matrix(kernel_function, x_list_1, x_list_2=None, save_path=None, thread_parallel=True, thread_jobs=4):
    """Build the Gram matrix associated with the given kernel
    Warning: with thread_parallel True sometimes the penylane QNode return [] instead of a float number
    """
    # check if only the input X is given or both Xtrain and Xtest
    if x_list_2 is None:
        x_list_2 = x_list_1

    # number of features must be equal
    assert x_list_1.shape[1] == x_list_2.shape[1], \
        "The second dimension (number of features) must be equal for both lists"

    # check dimension
    n, m = x_list_1.shape[0], x_list_2.shape[0]

    # i know, for Xtrain only the matrix is symmetric... but can we check it afterwards?
    def gram_row_run(i):
        # return [kernel_function(x_list_1[i], x_list_2[j]) for j in range(m)]
        l = []
        for j in range(m):
            itemm = kernel_function(x_list_1[i], x_list_2[j])
            item = None
            try:
                item = float(itemm)
            except Exception as e:
                print("Error ", e, "due to item", itemm, "in", i, x_list_1[i], j, x_list_2[j])
                exit(-1)
            l.append(item)
        print(".", end="")
        return l

    if thread_parallel:
        gram_matrix = Parallel(n_jobs=thread_jobs)(
            delayed(gram_row_run)(i) for i in range(n))
    else:
        gram_matrix = [gram_row_run(i) for i in range(n)]

    # from list to numpy
    gram_matrix = np.array(gram_matrix).T

    # save to file if specified
    if save_path is not None:
        np.savetxt(save_path, gram_matrix, delimiter=",")

    return gram_matrix


def build_gram_matrix_of_classical_kernels(X_train, X_test, experiment_folder):
    gaussian_gamma0_01_kernel = lambda x1, x2: rbf_kernel(x1.reshape(1, -1), x2.reshape(1, -1), gamma=.01)
    gaussian_gamma1_kernel = lambda x1, x2: rbf_kernel(x1.reshape(1, -1), x2.reshape(1, -1), gamma=1)
    gaussian_gamma100_kernel = lambda x1, x2: rbf_kernel(x1.reshape(1, -1), x2.reshape(1, -1), gamma=100)
    gaussian_gamma10000_kernel = lambda x1, x2: rbf_kernel(x1.reshape(1, -1), x2.reshape(1, -1), gamma=10000)
    build_gram_matrix(gaussian_gamma0_01_kernel, X_train, save_path=f"{experiment_folder}/gaussian_0_01_train.csv")
    build_gram_matrix(gaussian_gamma0_01_kernel, X_train, X_test, save_path=f"{experiment_folder}/gaussian_0_01_test.csv")
    build_gram_matrix(gaussian_gamma1_kernel, X_train, save_path=f"{experiment_folder}/gaussian_1_train.csv")
    build_gram_matrix(gaussian_gamma1_kernel, X_train, X_test, save_path=f"{experiment_folder}/gaussian_1_test.csv")
    build_gram_matrix(gaussian_gamma100_kernel, X_train, save_path=f"{experiment_folder}/gaussian_100_train.csv")
    build_gram_matrix(gaussian_gamma100_kernel, X_train, X_test, save_path=f"{experiment_folder}/gaussian_100_test.csv")
    build_gram_matrix(gaussian_gamma10000_kernel, X_train, save_path=f"{experiment_folder}/gaussian_10000_train.csv")
    build_gram_matrix(gaussian_gamma10000_kernel, X_train, X_test, save_path=f"{experiment_folder}/gaussian_10000_test.csv")

    laplacian_gamma0_01_kernel = lambda x1, x2: laplacian_kernel(x1.reshape(1, -1), x2.reshape(1, -1), gamma=.01)
    laplacian_gamma1_kernel = lambda x1, x2: laplacian_kernel(x1.reshape(1, -1), x2.reshape(1, -1), gamma=1)
    laplacian_gamma100_kernel = lambda x1, x2: laplacian_kernel(x1.reshape(1, -1), x2.reshape(1, -1), gamma=100)
    laplacian_gamma10000_kernel = lambda x1, x2: laplacian_kernel(x1.reshape(1, -1), x2.reshape(1, -1), gamma=10000)
    build_gram_matrix(laplacian_gamma0_01_kernel, X_train, save_path=f"{experiment_folder}/laplacian_0_01_train.csv")
    build_gram_matrix(laplacian_gamma0_01_kernel, X_train, X_test, save_path=f"{experiment_folder}/laplacian_0_01_test.csv")
    build_gram_matrix(laplacian_gamma1_kernel, X_train, save_path=f"{experiment_folder}/laplacian_1_train.csv")
    build_gram_matrix(laplacian_gamma1_kernel, X_train, X_test, save_path=f"{experiment_folder}/laplacian_1_test.csv")
    build_gram_matrix(laplacian_gamma100_kernel, X_train, save_path=f"{experiment_folder}/laplacian_100_train.csv")
    build_gram_matrix(laplacian_gamma100_kernel, X_train, X_test, save_path=f"{experiment_folder}/laplacian_100_test.csv")
    build_gram_matrix(laplacian_gamma10000_kernel, X_train, save_path=f"{experiment_folder}/laplacian_10000_train.csv")
    build_gram_matrix(laplacian_gamma10000_kernel, X_train, X_test, save_path=f"{experiment_folder}/laplacian_10000_test.csv")

    poly_d1_kernel = lambda x1, x2: polynomial_kernel(x1.reshape(1, -1), x2.reshape(1, -1), degree=1)
    poly_d2_kernel = lambda x1, x2: polynomial_kernel(x1.reshape(1, -1), x2.reshape(1, -1), degree=2)
    poly_d3_kernel = lambda x1, x2: polynomial_kernel(x1.reshape(1, -1), x2.reshape(1, -1), degree=3)
    poly_d4_kernel = lambda x1, x2: polynomial_kernel(x1.reshape(1, -1), x2.reshape(1, -1), degree=4)
    build_gram_matrix(poly_d1_kernel, X_train, save_path=f"{experiment_folder}/poly_d1_train.csv")
    build_gram_matrix(poly_d1_kernel, X_train, X_test, save_path=f"{experiment_folder}/poly_d1_test.csv")
    build_gram_matrix(poly_d2_kernel, X_train, save_path=f"{experiment_folder}/poly_d2_train.csv")
    build_gram_matrix(poly_d2_kernel, X_train, X_test, save_path=f"{experiment_folder}/poly_d2_test.csv")
    build_gram_matrix(poly_d3_kernel, X_train, save_path=f"{experiment_folder}/poly_d3_train.csv")
    build_gram_matrix(poly_d3_kernel, X_train, X_test, save_path=f"{experiment_folder}/poly_d3_test.csv")
    build_gram_matrix(poly_d4_kernel, X_train, save_path=f"{experiment_folder}/poly_d4_train.csv")
    build_gram_matrix(poly_d4_kernel, X_train, X_test, save_path=f"{experiment_folder}/poly_d4_test.csv")


def get_svm_metrics(gram_train, gram_test, y_train, y_test):
    """
    Train and test the kernel through the SVM algorithm
    :param gram_train: Gram matrix of the training set
    :param gram_test: Gram matrix of the test set
    :param y_train: labels of the training set
    :param y_test: labels of the test set
    :return: (clf, accuracy, precision, recall, fscore) where clf is the trained SVM, and the others are the respective metrics
    """

    clf = svm.SVC(kernel='precomputed')
    # training
    clf.fit(gram_train, y_train)
    # testing
    if len(gram_test.shape) == 1:  # if I have just one test sample... :(
        gram_test = gram_test.reshape(1, -1)
    y_pred = clf.predict(gram_test)
    # results
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
    return clf, accuracy, precision, recall, fscore


def get_metrics_for_all_kernels(y_train, y_test, experiment_folder, layers):

    KERNEL_LIST = [f"gaussian_{s}" for s in ["0_01", "1", "100", "10000"]] \
                  + [f"laplacian_{s}" for s in ["0_01", "1", "100", "10000"]] \
                  + [f"NTK_SHIRAI_layer_{i+1}" for i in range(layers)] \
                  + [f"PK_SHIRAI_layer_{i+1}" for i in range(layers)]

    df = pd.DataFrame(columns=["kernel", "accuracy", "precision", "recall", "fscore"])

    for kernel in KERNEL_LIST:
        gram_train = np.loadtxt(f"{experiment_folder}/{kernel}_train.csv", delimiter=",")
        gram_test = np.loadtxt(f"{experiment_folder}/{kernel}_test.csv", delimiter=",")
        _, accuracy, precision, recall, fscore = get_svm_metrics(gram_train, gram_test, y_train, y_test)
        df.loc[len(df)] = {
            "kernel": kernel,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "fscore": fscore
        }
        df.to_pickle(f"{experiment_folder}/statistics.pickle")

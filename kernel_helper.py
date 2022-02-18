from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KernelCenterer, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import svm
import numpy as np


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


def build_gram_matrix(kernel_function, x_list_1, x_list_2=None):
    """Build the Gram matrix associated with the given kernel"""
    # check if only the input X is given or both Xtrain and Xtest
    if x_list_2 is None:
        x_list_2 = x_list_1

    # number of features must be equal
    assert x_list_1.shape[1] == x_list_2.shape[1], "The second dimension must be equal (for IRIS must be 4)"

    # check dimension
    n, m = x_list_1.shape[0], x_list_2.shape[0]
    gram_matrix = np.zeros((n, m))

    # i know, for Xtrain only the matrix is symmetric... but can we check it afterwards?
    for i in range(n):
        for j in range(m):
            gram_matrix[i][j] = kernel_function(x_list_1[i], x_list_2[j])

    return gram_matrix


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
    y_pred = clf.predict(gram_test)
    # results
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
    return clf, accuracy, precision, recall, fscore

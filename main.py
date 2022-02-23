"""
Main file for noiseless simulation of NTK / Path Kernel
"""
import pandas as pd
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
from multiprocessing import Process, connection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config_loader import get_config
from kernel_helper import build_gram_matrix, get_svm_metrics
from pennylane_fixed_qubit_circuits import zz_kernel
from main_training import run_path_kernel_process


TRAIN_TEST_SPLIT_RANDOM_SEED = int(get_config("TRAIN_TEST_SPLIT_RANDOM_SEED"))
TEST_PERCENTAGE_OF_DATASET = float(get_config("TEST_PERCENTAGE_OF_DATASET"))
N_LAYERS = int(get_config("N_LAYERS"))


def main():
    print("The program is started")

    # load the dataset
    haberman_df = pd.read_pickle("downloaded_datasets/haberman.pickle")
    haberman_df = haberman_df[:3]
    y = haberman_df['target'].to_numpy()
    X = haberman_df.drop('target', axis=1).to_numpy()
    print("Dataset loaded")

    # preprocess dataset (normalize, choose a subset of items, etc...)
    X = MinMaxScaler().fit_transform(X)

    # split in training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PERCENTAGE_OF_DATASET, random_state=TRAIN_TEST_SPLIT_RANDOM_SEED)

    # calculate Gram Matrix associated with kernel ZZFeatureMap
    print("Calculating ZZ feature map kernel gram matrix (training set)")
    ZZ_kernel_train = build_gram_matrix(zz_kernel, X_train)
    pnp.savetxt("output/haberman/zz_kernel_matrices/ZZ_kernel_train.csv", ZZ_kernel_train, delimiter=",")

    print("Calculating ZZ feature map kernel gram matrix (testing set)")
    ZZ_kernel_test = build_gram_matrix(zz_kernel, X_test, X_train)
    pnp.savetxt("output/haberman/zz_kernel_matrices/ZZ_kernel_test.csv", ZZ_kernel_test, delimiter=",")

    # define and train on multiple processes - one process per layer of QNN
    print("Starting processes...")
    processes = [Process(
        target=run_path_kernel_process, args=(X_train, X_test, y_train, y_test, i+1)) for i in range(N_LAYERS)]
    # start all processes
    for i in range(N_LAYERS):
        processes[i].start()
    # wait until each process is in READY state (meaning it has terminated)
    for i in range(N_LAYERS):
        processes[i].join()
    print("Ended!")

    print("Starting accuracy checking with SVMs")
    _, ZZ_accuracy, _, _, _ = get_svm_metrics(ZZ_kernel_train, ZZ_kernel_test, y_train, y_test)
    print(f"SVM ZZ accuracy: {ZZ_accuracy}")

    for i in range(N_LAYERS):
        NTK_kernel_train = pnp.loadtxt(f"output/haberman/ntk_kernel_matrices/NTK_kernel_LAYER_{i}_train.csv", delimiter=",")
        NTK_kernel_test = pnp.loadtxt(f"output/haberman/ntk_kernel_matrices/NTK_kernel_LAYER_{i}_test.csv", delimiter=",")
        _, NTK_accuracy, _, _, _ = get_svm_metrics(NTK_kernel_train, NTK_kernel_test, y_train, y_test)
        print(f"SVM NTK accuracy with {i} layers: {NTK_accuracy}")

    for i in range(N_LAYERS):
        PK_kernel_train = pnp.loadtxt(f"output/haberman/path_kernel_matrices/PATH_kernel_LAYER_{i}_train.csv", delimiter=",")
        PK_kernel_test = pnp.loadtxt(f"output/haberman/path_kernel_matrices/PATH_kernel_LAYER_{i}_test.csv", delimiter=",")
        _, PK_accuracy, _, _, _ = get_svm_metrics(PK_kernel_train, PK_kernel_test, y_train, y_test)
        print(f"SVM PK accuracy with {i} layers: {PK_accuracy}")


if __name__ == '__main__':
    main()

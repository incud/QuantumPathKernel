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
from kernel_helper import build_gram_matrix
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


if __name__ == '__main__':
    main()

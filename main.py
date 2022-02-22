"""
Main file for noiseless simulation of NTK / Path Kernel
"""
import pandas as pd
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
from multiprocessing import Process, connection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from kernel_helper import build_gram_matrix
from pennylane_2qubit_circuits import zz_kernel, train_shirai_circuit, ntk_shirai_kernel_function, path_kernel_function


TRAIN_TEST_SPLIT_RANDOM_SEED = 43
TEST_PERCENTAGE_OF_DATASET = 0.33
N_LAYERS = 20


def run_shirai_circuit(X_train, X_test, y_train, y_test, n_layers):

    # run training of QNN
    opt = AdamOptimizer(0.5, beta1=0.9, beta2=0.999)
    save_path = f"output/haberman/training_trace/training_trace_LAYER_{n_layers}.pickle"
    training_df = train_shirai_circuit(X_train, X_test, y_train, y_test, opt, layers=n_layers, batch_size=20, epochs=1000, df_save_path=save_path)
    training_df.to_pickle(save_path)

    # create NTK matrix
    ntk_kernel = lambda x1, x2: ntk_shirai_kernel_function(x1, x2, training_df.iloc[len(training_df)-1])
    NTK_kernel_train = build_gram_matrix(ntk_kernel, X_train)
    pnp.savetxt(f"output/haberman/ntk_kernel_matrices/NTK_kernel_LAYER_{n_layers}_train.csv", NTK_kernel_train, delimiter=",")
    NTK_kernel_test = build_gram_matrix(ntk_kernel, X_test, X_train)
    pnp.savetxt(f"output/haberman/ntk_kernel_matrices/NTK_kernel_LAYER_{n_layers}_test.csv", NTK_kernel_test, delimiter=",")

    # create PATH K. matrix
    path_kernel = lambda x1, x2: path_kernel_function(x1, x2, training_df.iloc[len(training_df)-1])
    PK_kernel_train = build_gram_matrix(path_kernel, X_train)
    pnp.savetxt(f"output/haberman/path_kernel_matrices/PATH_kernel_LAYER_{n_layers}_train.csv", PK_kernel_train, delimiter=",")
    PK_kernel_test = build_gram_matrix(path_kernel, X_test, X_train)
    pnp.savetxt(f"output/haberman/path_kernel_matrices/PATH_kernel_LAYER_{n_layers}_test.csv", PK_kernel_test, delimiter=",")


if __name__ == '__main__':

    # load the dataset
    haberman_df = pd.read_pickle("downloaded_datasets/haberman.pickle")
    haberman_df = haberman_df[:3]
    y = haberman_df.drop('target', axis=1).to_numpy()
    X = haberman_df.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PERCENTAGE_OF_DATASET, random_state=TRAIN_TEST_SPLIT_RANDOM_SEED)

    # preprocess dataset (normalize, choose a subset of items, etc...)
    X = normalize(X, axis=0, norm=max)

    # calculate Gram Matrix associated with kernel ZZFeatureMap
    ZZ_kernel_train = build_gram_matrix(zz_kernel, X_train)
    pnp.savetxt("output/haberman/zz_kernel_matrices/ZZ_kernel_train.csv", ZZ_kernel_train, delimiter=",")
    ZZ_kernel_test = build_gram_matrix(zz_kernel, X_test, X_train)
    pnp.savetxt("output/haberman/zz_kernel_matrices/ZZ_kernel_test.csv", ZZ_kernel_test, delimiter=",")

    # define and train on multiple processes
    processes = [Process(
        target=run_shirai_circuit, args=(X_train, X_test, y_train, y_test, i+1)) for i in range(N_LAYERS)]
    # start all processes
    for i in range(N_LAYERS):
        processes[i].start()
    # wait until each process is in READY state (meaning it has terminated)
    connection.wait(p.sentinel for p in processes)

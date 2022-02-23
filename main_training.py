"""
Main file for noiseless simulation of NTK / Path Kernel
"""
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
from kernel_helper import build_gram_matrix
from pennylane_fixed_qubit_circuits import train_shirai_circuit, ntk_shirai_kernel_function, path_kernel_function


def run_path_kernel_process(X_train, X_test, y_train, y_test, n_layers, optimizer_str, batch_size, epochs, experiment_folder):

    # choose an optimizer (code injection in three, two, one...)
    opt = eval(optimizer_str)

    # start training of QNN and save trace of training to file
    training_path = f"{experiment_folder}/QNN_SHIRAI_layer_{n_layers}_training_trace.pickle"
    training_df = train_shirai_circuit(
        X_train, X_test, y_train, y_test, opt,
        layers=n_layers, batch_size=batch_size, epochs=epochs, df_save_path=training_path)
    training_df.to_pickle(training_path)

    # create NTK matrix
    ntk_kernel = lambda x1, x2: ntk_shirai_kernel_function(x1, x2, training_df.iloc[len(training_df)-1])
    build_gram_matrix(ntk_kernel, X_train, save_path=f"{experiment_folder}/NTK_SHIRAI_layer_{n_layers}_train.csv")
    build_gram_matrix(ntk_kernel, X_train, X_test, save_path=f"{experiment_folder}/NTK_SHIRAI_layer_{n_layers}_test.csv")

    # create PATH K. matrix
    path_kernel = lambda x1, x2: path_kernel_function(x1, x2, training_df)
    build_gram_matrix(path_kernel, X_train, save_path=f"{experiment_folder}/PK_SHIRAI_layer_{n_layers}_train.csv")
    build_gram_matrix(path_kernel, X_test, X_train, save_path=f"{experiment_folder}/PK_SHIRAI_layer_{n_layers}_test.csv")

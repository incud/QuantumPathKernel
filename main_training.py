"""
Main file for noiseless simulation of NTK / Path Kernel
"""
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
from kernel_helper import build_gram_matrix
from pennylane_fixed_qubit_circuits import train_shirai_circuit, ntk_shirai_kernel_function, path_kernel_function
from config_loader import get_config

EPOCHS = int(get_config("EPOCHS"))


def run_path_kernel_process(X_train, X_test, y_train, y_test, n_layers):

    # run training of QNN
    opt = AdamOptimizer(0.5, beta1=0.9, beta2=0.999)
    save_path = f"output/haberman/training_trace/training_trace_LAYER_{n_layers}.pickle"

    print(f"Started training of {n_layers} layers QNN")
    training_df = train_shirai_circuit(
        X_train, X_test, y_train, y_test, opt, layers=n_layers, batch_size=20, epochs=EPOCHS, df_save_path=save_path)
    training_df.to_pickle(save_path)

    # create NTK matrix
    ntk_kernel = lambda x1, x2: ntk_shirai_kernel_function(x1, x2, training_df.iloc[len(training_df)-1])

    print(f"Calculating NTK kernel gram matrix (training set) of {n_layers} layers QNN")
    NTK_kernel_train = build_gram_matrix(ntk_kernel, X_train)
    pnp.savetxt(f"output/haberman/ntk_kernel_matrices/NTK_kernel_LAYER_{n_layers}_train.csv", NTK_kernel_train, delimiter=",")

    print(f"Calculating NTK kernel gram matrix (testing set) of {n_layers} layers QNN")
    NTK_kernel_test = build_gram_matrix(ntk_kernel, X_test, X_train)
    pnp.savetxt(f"output/haberman/ntk_kernel_matrices/NTK_kernel_LAYER_{n_layers}_test.csv", NTK_kernel_test, delimiter=",")

    # create PATH K. matrix
    path_kernel = lambda x1, x2: path_kernel_function(x1, x2, training_df)

    print(f"Calculating PATH K. kernel gram matrix (training set) of {n_layers} layers QNN")
    PK_kernel_train = build_gram_matrix(path_kernel, X_train)
    pnp.savetxt(f"output/haberman/path_kernel_matrices/PATH_kernel_LAYER_{n_layers}_train.csv", PK_kernel_train, delimiter=",")

    print(f"Calculating PATH K. kernel gram matrix (training set) of {n_layers} layers QNN")
    PK_kernel_test = build_gram_matrix(path_kernel, X_test, X_train)
    pnp.savetxt(f"output/haberman/path_kernel_matrices/PATH_kernel_LAYER_{n_layers}_test.csv", PK_kernel_test, delimiter=",")

    print(f"Ended {n_layers} layers QNN")

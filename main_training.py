"""
Main file for noiseless simulation of NTK / Path Kernel
"""
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
from kernel_helper import build_gram_matrix
from pennylane_fixed_qubit_circuits import PathKernelSimulator


def run_path_kernel_process(X_train, X_test, y_train, y_test, n_layers, optimizer_str, batch_size, epochs, experiment_folder):

    # choose an optimizer (code injection in three, two, one...)
    opt = eval(optimizer_str)

    # generate instance of PathKernelGenerator
    n_qubits = X_train.shape[1]
    print(f"{n_layers}: The number of qubit matches the number of features ({n_qubits})")
    sim = PathKernelSimulator(n_qubits)

    # start training of QNN and save trace of training to file
    print(f"{n_layers}: Started training")
    training_path = f"{experiment_folder}/QNN_SHIRAI_layer_{n_layers}_training_trace.pickle"
    training_df = sim.train_shirai_circuit(
        X_train, X_test, y_train, y_test, opt,
        layers=n_layers, batch_size=batch_size, epochs=epochs, df_save_path=training_path)
    training_df.to_pickle(training_path)

    # create NTK matrix
    ntk_kernel = lambda x1, x2: sim.ntk_shirai_kernel_function(x1, x2, training_df.iloc[len(training_df)-1])
    print(f"{n_layers}: Calculating NTK (train)")
    build_gram_matrix(ntk_kernel, X_train, save_path=f"{experiment_folder}/NTK_SHIRAI_layer_{n_layers}_train.csv")
    print(f"{n_layers}: Calculating NTK (test)")
    build_gram_matrix(ntk_kernel, X_train, X_test, save_path=f"{experiment_folder}/NTK_SHIRAI_layer_{n_layers}_test.csv")

    # create PATH K. matrix
    path_kernel = lambda x1, x2: sim.path_kernel_function(x1, x2, training_df)
    print(f"{n_layers}: Calculating PK (train)")
    build_gram_matrix(path_kernel, X_train, save_path=f"{experiment_folder}/PK_SHIRAI_layer_{n_layers}_train.csv")
    print(f"{n_layers}: Calculating PK (test)")
    build_gram_matrix(path_kernel, X_test, X_train, save_path=f"{experiment_folder}/PK_SHIRAI_layer_{n_layers}_test.csv")

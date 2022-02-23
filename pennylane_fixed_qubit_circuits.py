"""
For now the circuit are specified only for N=xxx qubits. If there is the need to generalize to N=3, 4, 5, ... qubits
it might be interesting to create a class that gets N as input and instantiate the devices, functions etc.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane_circuits import ZZFeatureMap, ShiraiLayerAnsatz
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import pandas as pd
from joblib import Parallel, delayed

N_QUBITS = 3
device_fixed_qubits = qml.device("default.qubit", wires=N_QUBITS)
projector = pnp.zeros((2**N_QUBITS, 2**N_QUBITS))
projector[0, 0] = 1


@qml.qnode(device_fixed_qubits)
def zz_kernel(x1, x2):
    ZZFeatureMap(x1, reps=1, wires=range(N_QUBITS))
    qml.adjoint(ZZFeatureMap)(x2, reps=1, wires=range(N_QUBITS))
    return qml.expval(qml.Hermitian(projector, wires=range(N_QUBITS)))


@qml.qnode(device_fixed_qubits)
def shirai_circuit(x, theta, layers):
    assert theta.shape == (layers, 3, 3), "Theta shape must be ({},3,3), now it is {}".format(layers, theta.shape)
    # quantum feature map (Havlicek)
    ZZFeatureMap(x, range(N_QUBITS), reps=1)
    # variational form
    for i in range(layers):
        # Shirai's ansatz
        ShiraiLayerAnsatz(theta[i], range(N_QUBITS))
    # measurement - just the last qubit
    return qml.expval(qml.PauliZ(N_QUBITS-1))


@qml.qnode(device_fixed_qubits)
def datareup_circuit(x, theta, layers):
    for l in range(layers):
        # quantum feature map (Havlicek)
        ZZFeatureMap(x, range(N_QUBITS), reps=1)
        # Shirai's ansatz
        ShiraiLayerAnsatz(theta[l], range(N_QUBITS))
    # measurement  - just the last qubit
    return qml.expval(qml.PauliZ(N_QUBITS-1))


_shirai_circuit_gradient = qml.gradients.param_shift(shirai_circuit)


def shirai_circuit_gradient():
    return _shirai_circuit_gradient


def iterate_minibatches(inputs, targets, batch_size):
    """A generator for batches of the input data"""
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]


def train_shirai_circuit(X_train, X_test, y_train, y_test, opt, layers=1, batch_size=20, epochs=1000, df_save_path=None):
    """
    Run training session and store the trace at each computational step
    :param X_train: data training set
    :param X_test: data testing set
    :param y_train: label training set
    :param y_test: label testing set
    :param opt: optimizer e.g. AdamOptimizer(0.5, beta1=0.9, beta2=0.999)
    :param layers: number of layers
    :param batch_size: batch size
    :param epochs: number of epochs
    :param df_save_path: pandas save path
    :return: the pandas dataframe with the trace of the training process
    """

    # setup
    assert opt is not None, "Please define an optimizer such as AdamOptimizer(0.5, beta1=0.9, beta2=0.999)"

    df = pd.DataFrame(
        columns=['this_epoch', 'this_params', 'this_cost', 'this_train_accuracy', 'this_test_accuracy',
                 'batch_size', 'layers', 'epochs'])

    def get_cost_item(the_params, xi, yi):
        return (shirai_circuit(xi, the_params, layers) - yi)**2

    def get_cost_batch(the_params, x_batch, y_batch):
        return sum([get_cost_item(the_params, xi, yi) for (xi, yi) in zip(x_batch, y_batch)]) / len(y_batch)

    def get_cost(the_params):
        return get_cost_batch(the_params, X_train, y_train)

    def get_accuracy_item(the_params, xi, yi):
        return pnp.isclose(pnp.sign(shirai_circuit(xi, the_params, layers)), yi)

    def get_accuracy(the_params, X_set, y_set):
        return sum([get_accuracy_item(the_params, xi, yi) for (xi, yi) in zip(X_set, y_set)]) / len(y_set)

    def create_trace_row(the_params, the_epoch):
        return {
            'this_epoch': the_epoch,
            'this_params': the_params,
            'this_cost': get_cost(the_params),
            'this_train_accuracy': get_accuracy(the_params, X_train, y_train),
            'this_test_accuracy': get_accuracy(the_params, X_test, y_test),
            'batch_size': batch_size,
            'layers': layers,
            'epochs': epochs
        }

    # generate initial thetas from gaussian distribution mean=0 var=1
    params = pnp.random.uniform(size=(layers, N_QUBITS, 3))

    # save the init configuration
    df.loc[len(df)] = create_trace_row(params, 0)

    # start training
    for i in range(epochs):

        # optimization step over a batch of elements
        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size=batch_size):
            params = opt.step(lambda v: get_cost_batch(v, X_batch, y_batch), params)

        # calculate the efficacy of the solution at the current step
        df.loc[len(df)] = create_trace_row(params, i+1)

        # backup save at the given path
        if i % 10 == 0 and df_save_path is not None:
            df.to_pickle(df_save_path)

    return df


def wrap_grad(data):
    """Wrap an array of data, marking it as differentiable"""
    return pnp.array(data, requires_grad=True)


def wrap_no_grad(data):
    """Wrap an array of data, marking it as NON differentiable"""
    return pnp.array(data, requires_grad=False)


def linearized_shirai_kernel_function(x1, x2, gradient, params):
    """
    Calculate linearized kernel over Shirai's circuit
    :param x1: first sample
    :param x2: second sample
    :param gradient: pre-computed gradient
    :param params: actual theta params
    :return: grad_theta(x1, theta) * grad_theta(x2, theta)
    """
    assert len(params.shape) == 3, f"Params must be a 3D array of shape N_LAYERS * 3 * 3 (it is {params.shape})"
    n_layer = params.shape[0]
    g1 = gradient(wrap_no_grad(x1), params, wrap_no_grad(n_layer)).reshape(-1)
    g2 = gradient(wrap_no_grad(x2), params, wrap_no_grad(n_layer)).reshape(-1)
    return g1.dot(g2)


def ntk_shirai_kernel_function(x1, x2, last_training_row):
    """
    Calculate the NTK kernel between sample x1 and x2
    :param x1: first sample
    :param x2: second sample
    :param last_training_row: last row of training dataframe
    :return: NTK between x1 and x2
    """

    gradient = shirai_circuit_gradient()
    params = last_training_row['this_params']
    return linearized_shirai_kernel_function(x1, x2, gradient, params)


def path_kernel_function(x1, x2, training_df, thread_parallel=False, thread_jobs=16):
    """
    Calculate the NTK kernel between sample x1 and x2
    :param x1: first sample
    :param x2: second sample
    :param training_df: training trace dataframe
    :param thread_parallel: true in order to use "joblib" library for parallel thread-based execution
    :param thread_jobs: if thread_parallel is True than indicates the number of threads, otherwise ignore
    :return: PATH KERNEL between x1 and x2
    """
    gradient = shirai_circuit_gradient()
    params_epochs = training_df['this_params']
    if thread_parallel:
        contrib_run = lambda params: linearized_shirai_kernel_function(x1, x2, gradient, params)
        contributions = Parallel(n_jobs=thread_jobs, prefer="threads")(
            delayed(contrib_run)(params) for params in params_epochs)
    else:
        contributions = [linearized_shirai_kernel_function(x1, x2, gradient, params) for params in params_epochs]

    return pnp.mean(contributions)

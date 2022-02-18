"""
For now the circuit are specified only for N=2 qubits. If there is the need to generalize to N=3, 4, 5, ... qubits
it might be interesting to create a class that gets N as input and instantiate the devices, functions etc.
"""
import pennylane as qml
from pennylane import numpy as pnp
from pennylane_circuits import ZZFeatureMap, ShiraiLayerAnsatz

N_QUBITS = 2
device_2qubits = qml.device("default.qubit", wires=N_QUBITS)
projector = pnp.zeros((2**N_QUBITS, 2**N_QUBITS))
projector[0, 0] = 1


@qml.qnode(device_2qubits)
def zz_kernel(x1, x2):
    ZZFeatureMap(x1, reps=1, wires=range(N_QUBITS))
    qml.adjoint(ZZFeatureMap)(x2, reps=1, wires=range(N_QUBITS))
    return qml.expval(qml.Hermitian(projector, wires=range(N_QUBITS)))


@qml.qnode(device_2qubits)
def shirai_circuit(x, theta, layers):
    # quantum feature map (Havlicek)
    ZZFeatureMap(x, 1, range(N_QUBITS))
    for l in range(layers):
        # Shirai's ansatz
        ShiraiLayerAnsatz(theta[l], range(N_QUBITS))
    # measurement - just the last qubit
    return qml.expval(qml.PauliZ(N_QUBITS-1))


@qml.qnode(device_2qubits)
def datareup_circuit(x, theta, layers):
    for l in range(layers):
        # quantum feature map (Havlicek)
        ZZFeatureMap(x, 1, range(N_QUBITS))
        # Shirai's ansatz
        ShiraiLayerAnsatz(theta[l], range(N_QUBITS))
    # measurement  - just the last qubit
    return qml.expval(qml.PauliZ(N_QUBITS-1))

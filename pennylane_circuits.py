import pennylane as qml
from pennylane import numpy as pnp


def ZZFeatureMap(x, wires, reps=1):
    """Havlicek-like mapping, using Z and ZZ Pauli only
    x: number of data (1 data per wire)
    wires: the list of wires on which the circuit acts
    reps: number of repetitions (>= 1)
    """
    assert len(x) == len(wires), "One data per qubit ({})".format(len(x))
    assert reps >= 1, "At least one repetition is needed"
    n = len(x)
    for _ in range(reps):
        for i in range(n):
            qml.Hadamard(wires=wires[i])
            qml.RZ(2 * x[i], wires=wires[i])
        for i in range(n):
            for j in range(i, n):
                if i != j:
                    qml.CNOT(wires=(wires[i], wires[j]))
                    qml.RZ(2 * (pnp.pi - x[i]) * (pnp.pi - x[j]), wires=wires[j])
                    qml.CNOT(wires=(wires[i], wires[j]))


def ShiraiLayerAnsatz(theta, wires):
    """The ansatz I suppose was used in Shirai's work(https://arxiv.org/abs/2111.02951)
    theta: WIRES x 3 array of parameters
    wires: the list of wires on which the circuit acts
    """
    assert theta.shape == (len(wires), 3), "Theta must have size {}".format((len(wires), 3))
    n = len(wires)
    for i in range(n):
        j = (i + 1) % n
        qml.IsingXX(theta[i][0], wires=(wires[i], wires[j]))
        qml.IsingYY(theta[i][1], wires=(wires[i], wires[j]))
        qml.IsingZZ(theta[i][2], wires=(wires[i], wires[j]))

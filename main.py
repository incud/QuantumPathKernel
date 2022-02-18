from multiprocessing import Process
import os

import pandas as pd

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def quantum_circ(x, theta):
    qml.RX(x, wires=0)
    qml.RZ(theta, wires=0)
    return qml.expval(qml.PauliZ(0))


g = qml.gradients.param_shift(quantum_circ)


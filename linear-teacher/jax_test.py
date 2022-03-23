import jax
import pennylane as qml
from pennylane.optimize import AdamOptimizer

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev, interface="jax")
def circuit(param):
    qml.RX(param, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


grad_circuit = jax.grad(circuit)
optimizer = AdamOptimizer()
params = 0.123

for i in range(4):
    params = optimizer.step(circuit, params, grad_fn=grad_circuit)

import pennylane as qml
import pennylane.numpy as pnp
import pennylane.tape

n_qubits = 2
cir_depth = 10
pnp.random.seed(1)
dev = qml.device('default.qubit', wires=n_qubits+1)
param = pnp.random.uniform(0.0001, 2*pnp.pi, cir_depth*n_qubits)

@qml.qnode(dev)
def cost_fn(weights):
    for i in range(n_qubits):
        qml.RY(pnp.pi/4, wires=i)
    for j in range(cir_depth):
        if j % 2 == 0:
            for i in range(n_qubits):
                qml.RX(weights[i + n_qubits*j], wires=i)
        else:
            for i in range(n_qubits):
                qml.RY(weights[i + n_qubits*j], wires=i)
        qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

mt_fn = qml.metric_tensor(cost_fn, approx=None)
mt = mt_fn(param)
eigvals, eigvecs = pnp.linalg.eigh(mt)


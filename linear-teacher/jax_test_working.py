# Added to silence some warnings.
from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer

dev = qml.device("default.qubit", wires=2)

@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(param):
    # These two gates represent our QML model.
    qml.RX(param, wires=0)
    qml.CNOT(wires=[0, 1])

    # The expval here will be the "cost function" we try to minimize.
    # Usually, this would be defined by the problem we want to solve,
    # but for this example we'll just use a single PauliZ.
    return qml.expval(qml.PauliZ(0))


def loss_fn(params):
    return circuit(params) - 1


grad_fn = jax.grad(loss_fn)
optimizer = AdamOptimizer()
params = 0.56754

print(f"Value: {params}")
for i in range(10):
    params -= grad_fn(params)
    print(f"Value: {params}")

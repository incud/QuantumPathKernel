import numpy as np
import optax
from mlp_classifier import RandomKernelClassifier, NeuralNetworkClassifier, DeepLinearClassifier, DeepNonLinearClassifier

N = 500
NUM_FEATURES = 5
NUM_CLASSES = 2
X = np.random.normal(size=(N, NUM_FEATURES))
y = np.random.randint(0, 2, size=(N,))

linear_clf = RandomKernelClassifier(n_hidden_nodes=100, num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
nonlinear_clf = NeuralNetworkClassifier(n_hidden_nodes=25, num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)

deep_linear_clf = DeepLinearClassifier(hidden_nodes=[10, 8, 4], num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
deep_nonlinear_clf = DeepNonLinearClassifier(hidden_nodes=[10, 8, 4], num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)

linear_clf.fit(X, y)
nonlinear_clf.fit(X, y)
deep_linear_clf.fit(X, y)
deep_nonlinear_clf.fit(X, y)

linear_score = linear_clf.score(X, y)
nonlinear_score = nonlinear_clf.score(X, y)
print(f"Linear {linear_score}, Non Linear {nonlinear_score}")
print(f"# param Linear {linear_clf.get_num_parameters()}, Non Linear {nonlinear_clf.get_num_parameters()}")

deep_linear_score = deep_linear_clf.score(X, y)
deep_nonlinear_score = deep_nonlinear_clf.score(X, y)
print(f"Deep Linear {deep_linear_score}, Non Linear {deep_nonlinear_score}")
print(f"# param Deep Linear {deep_linear_clf.get_num_parameters()}, Non Linear {deep_nonlinear_clf.get_num_parameters()}")

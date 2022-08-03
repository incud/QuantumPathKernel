import numpy as np
import optax
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mlp_classifier import NeuralNetworkClassifier
from path_kernel import GradientKernel
from plot_decision_boundaries import plot_decision_boundaries

np.random.seed(12345)
N = 500
NUM_FEATURES = 2
NUM_CLASSES = 2
X, y = make_classification(n_features=NUM_FEATURES, n_classes=NUM_CLASSES, n_redundant=0, n_informative=2, n_clusters_per_class=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

nn_clf = NeuralNetworkClassifier(n_hidden_nodes=25, num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
nn_clf.fit(X_train, y_train)
print(f"NN Score: {nn_clf.score(X_test, y_test)}")
plot_decision_boundaries(nn_clf, X_train, y_train, "nn_dec_bounds.png")

svm_merdosa_clf = SVC()
svm_merdosa_clf.fit(X_train, y_train)
print(f"SVM Merdosa Score: {svm_merdosa_clf.score(X_test, y_test)}")
plot_decision_boundaries(svm_merdosa_clf, X_train, y_train, "svm_merdosa_dec_bounds.png")

# nn_path_kernel_train = GradientKernel(nn_clf).path_kernel(X_train, X_train)
# nn_path_kernel_test = GradientKernel(nn_clf).path_kernel(X_test, X_train)
svm_clf = SVC(kernel=lambda test, train: GradientKernel(nn_clf).path_kernel(test, train))
svm_clf.fit(X_train, y_train)
print(f"SVM Score: {svm_clf.score(X_test, y_test)}")
plot_decision_boundaries(svm_clf, X_train, y_train, "svm_pk_dec_bounds_50.png")


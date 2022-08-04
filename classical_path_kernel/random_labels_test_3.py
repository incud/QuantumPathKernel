import numpy as np
import optax
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from mlp_classifier import DeepLinearClassifier, DeepNonLinearClassifier
from path_kernel import PathKernelClassifier
from plot_decision_boundaries import plot_decision_boundaries, plot_decision_areas

np.random.seed(22345)
DIR = "random_labels_test_3_images"
N = 30
NUM_FEATURES = 2
NUM_CLASSES = 2
X = MinMaxScaler().fit_transform(np.random.uniform(low=-1, high=1, size=(N, NUM_FEATURES)))
y = np.random.randint(0, 2, size=(N,))

optimizer = optax.adam(1e-3)
epochs = 10_000
nn = DeepNonLinearClassifier(hidden_nodes=[3, 3, 3], num_classes=NUM_CLASSES, optimizer=optimizer, epochs=epochs, seed=12345)
nn.fit(X, y)
plot_decision_boundaries(nn, X, y, f"{DIR}/dec_bounds_model", precision=0.01)
plot_decision_areas(None, X, y, load_path=f"{DIR}/dec_bounds_model", save_path=f"{DIR}/dec_area_model", precision=0.01)

for sample_frequency in [1000, 500, 200, 100, 50, 20, 10, 5, 1]:
    pk_clf = PathKernelClassifier(nn, sample_frequency=sample_frequency)
    pk_clf.fit(X, y)
    path_boundary = f"{DIR}/dec_bounds_pk_s{sample_frequency}"
    path_area = f"{DIR}/dec_area_pk_s{sample_frequency}"
    plot_decision_boundaries(pk_clf, X, y, path_boundary, precision=0.01)
    plot_decision_areas(None, X, y, load_path=path_boundary, save_path=path_area, precision=0.01)

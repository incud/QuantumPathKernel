import numpy as np
import optax
from mlp_classifier import DeepLinearClassifier, DeepNonLinearClassifier
from path_kernel import GradientKernel
from sklearn.svm import SVC
from plot_decision_boundaries import plot_decision_boundaries, plot_decision_areas


# np.random.seed(223432)
np.random.seed(22346)

DIR = "random_labels_test_4_images"
N = 5
N_RANGE = np.linspace(-0.8, 0.8, N)
NUM_FEATURES = 2
NUM_CLASSES = 2
X = np.array([[a, b] for a in N_RANGE for b in N_RANGE])
y = np.random.randint(0, 2, size=(N**2,))

optimizer = optax.adam(1e-3)
epochs = 10_000
nn = DeepNonLinearClassifier(hidden_nodes=[3, 3, 2], num_classes=NUM_CLASSES, optimizer=optimizer, epochs=epochs, seed=12345)
nn.fit(X, y)
plot_decision_boundaries(nn, X, y, f"{DIR}/dec_bounds_model", precision=0.025)
plot_decision_areas(None, X, y, load_path=f"{DIR}/dec_bounds_model", save_path=f"{DIR}/dec_area_model", precision=0.025)

path_kernel = GradientKernel(nn)
cache_training = [None] * epochs
cache_grid = [None] * epochs

for sample_frequency in [1000, 500, 200, 100, 50, 20, 10, 5, 1]:
    # set current sample frequency
    path_kernel.set_sample_frequency(sample_frequency)
    # create and train kernel
    pk_clf = SVC(kernel='precomputed')
    gram_x = path_kernel.path_kernel_cached(X, cache=cache_training)
    pk_clf.fit(X, y)
    # create and train areas
    path_boundary = f"{DIR}/dec_bounds_pk_s{sample_frequency}"
    path_area = f"{DIR}/dec_area_pk_s{sample_frequency}"
    plot_decision_boundaries(pk_clf, X, y, path_boundary, precision=0.025)
    plot_decision_areas(None, X, y, load_path=path_boundary, save_path=path_area, precision=0.025)

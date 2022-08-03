import numpy as np
import optax
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from mlp_classifier import DeepLinearClassifier, DeepNonLinearClassifier
from path_kernel import PathKernelClassifier
from plot_decision_boundaries import plot_decision_boundaries

np.random.seed(12345)
N = 100
NUM_FEATURES = 2
NUM_CLASSES = 2
X = MinMaxScaler().fit_transform(np.random.normal(size=(N, NUM_FEATURES)))
y = np.random.randint(0, 2, size=(N,))

nn_1 = DeepNonLinearClassifier(hidden_nodes=[5],       num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
nn_2 = DeepNonLinearClassifier(hidden_nodes=[5, 3],    num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
nn_3 = DeepNonLinearClassifier(hidden_nodes=[5, 3, 3], num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
lin_1 = DeepLinearClassifier(hidden_nodes=[5],         num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
lin_2 = DeepLinearClassifier(hidden_nodes=[5, 3],      num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
lin_3 = DeepLinearClassifier(hidden_nodes=[5, 3, 3],   num_classes=NUM_CLASSES, optimizer=optax.adam(1e-3), epochs=1000, seed=12345)
clfs = [nn_1, nn_2, nn_3, lin_1, lin_2, lin_3]
clf_titles = ["NN_h10", "NN_h10_h7", "NN_h10_h7_h3", "LIN_h10", "LIN_h10_h7", "LIN_h10_h7_h3"]
for clf, title in zip(clfs, clf_titles):
    clf.fit(X, y)
    plot_decision_boundaries(clf, X, y, f"random_labels_test_1_images/{title}/dec_bounds_model.png")

    # pk_200_clf = PathKernelClassifier(clf, sample_frequency=200)
    # pk_200_clf.fit(X, y)
    # plot_decision_boundaries(pk_200_clf, X, y, f"random_labels_test_1_images/{title}/dec_bounds_pk_s{200}.png")
#
    # pk_100_clf = PathKernelClassifier(clf, sample_frequency=100)
    # pk_100_clf.fit(X, y)
    # plot_decision_boundaries(pk_100_clf, X, y, f"random_labels_test_1_images/{title}/dec_bounds_pk_s{100}.png")
#
    # pk_50_clf = PathKernelClassifier(clf, sample_frequency=50)
    # pk_50_clf.fit(X, y)
    # plot_decision_boundaries(pk_50_clf, X, y, f"random_labels_test_1_images/{title}/dec_bounds_pk_s{50}.png")
#
    # pk_20_clf = PathKernelClassifier(clf, sample_frequency=20)
    # pk_20_clf.fit(X, y)
    # plot_decision_boundaries(pk_20_clf, X, y, f"random_labels_test_1_images/{title}/dec_bounds_pk_s{20}.png")

    pk_5_clf = PathKernelClassifier(clf, sample_frequency=5)
    pk_5_clf.fit(X, y)
    plot_decision_boundaries(pk_5_clf, X, y, f"random_labels_test_1_images/{title}/dec_bounds_pk_s{5}.png")

import numpy as np
import matplotlib.pyplot as plt #do not remove
from jax import numpy as jnp

def generate_sample_data(num_classes, n_samples):
    # Generate some data
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=n_samples, centers=num_classes,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting

    # Plot the data with K Means Labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(num_classes, random_state=0)
    labels = kmeans.fit(X).predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
    # plt.show()

    return jnp.array(X), labels

def get_training_set(X, labels, num_classes_learning):
    X_learning = X[np.where(labels < num_classes_learning)]
    labels_learning = labels[np.where(labels < num_classes_learning)]
    # plt.scatter(X_learning[:, 0], X_learning[:, 1], c=labels_learning, s=40, cmap='viridis');
    # plt.show()

    return X_learning, labels_learning, num_classes_learning


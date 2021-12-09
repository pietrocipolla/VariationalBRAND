from jax import numpy as jnp

def generate_some_data_example():
    #example : 5, 2d gaussians
    num_clusters = 5
    num_samples = 500

    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=num_samples, centers=num_clusters,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    return jnp.array(X)


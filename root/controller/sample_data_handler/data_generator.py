from jax import numpy as jnp
import numpy as np

def generate_some_data_example():
    #example : 5, 2d gaussians
    #centers = 5

    # num_samples = 1000
    small = [350, 250, 250, 49, 50, 50, 1]
    not_small = [200, 200, 250, 90, 10]
    num_samples = not_small

    centers = [[-5, 5], [-4, -4], [4, 4], [-0, 0],[-10, -10]]

    #cluster_std = [1, 1.4,1.4,1,0.1 ]

    cluster_std = 0.6

    # list(
    #     matrix(c(1, .9, .9, 1), 2, 2),
    #     diag(2),
    #     diag(2),
    #     matrix(c(1, -.75, -.75, 1), 2, 2),
    #     matrix(c(1, .9, .9, 1), 2, 2),
    #     matrix(c(1, -.9, -.9, 1), 2, 2),
    #     diag(.01, 2)
    # )

    # num_samples =

    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=num_samples, centers=centers,
                           cluster_std=cluster_std, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    return jnp.array(X)


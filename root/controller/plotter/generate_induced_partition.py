import matplotlib
from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters
import numpy as np


def generate_induced_partition(Y, robust_mean, hyperparameters_model: HyperparametersModel, variational_parameters: VariationalParameters):
    import matplotlib.pyplot as plt
    from jax import numpy as jnp
    ll = []
    for i in range(Y.shape[0]):
        ll.append(jnp.argmax(variational_parameters.phi_m_k[i, :]))


    print('Clusters\' numerosity')
    unique_clusters = np.unique(np.array(ll))
    num_clusters = len(unique_clusters)

    for i in unique_clusters:
        print('cluster ', i, ': ',ll.count(i))

    plt.scatter(Y[:, 0], Y[:, 1], c=[matplotlib.cm.get_cmap("Spectral")(float(i) / num_clusters) for i in ll])


    print(robust_mean)
    for i in range(hyperparameters_model.J):
        plt.scatter(robust_mean[i][0], robust_mean[i][1], color='red')
    for i in unique_clusters:
        plt.scatter(variational_parameters.nIW.mu[i, 0], variational_parameters.nIW.mu[i, 1], color='black')
    # plt.show()
    plt.savefig('figure.png')
    plt.close()
    print("\n\nPLOT available in /content/VariationalBRAND/tests/figure.png")

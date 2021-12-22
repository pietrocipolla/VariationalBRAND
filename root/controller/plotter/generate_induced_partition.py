import matplotlib

from controller.plotter.plot_covariance_ellipses import  plot_cov_ellipse
from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters
import numpy as np


def generate_induced_partition(Y, robust_mean, hyperparameters_model: HyperparametersModel, variational_parameters: VariationalParameters, cov_ellipse):
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
        if(hyperparameters_model.p == 2):
            print(variational_parameters.nIW.phi[i],variational_parameters.nIW.mu[i, :] )
            if cov_ellipse:
                plot_cov_ellipse(variational_parameters.nIW.mu[i],
                                 variational_parameters.nIW.phi[i]/(variational_parameters.nIW.nu[i] - hyperparameters_model.p -1))


    figure_name = 'figure'
    figure_filetype = '.png'
    if(cov_ellipse):
        figure_name = figure_name + '-ellipses' + figure_filetype
    else:
        figure_name = figure_name + figure_filetype

    plt.savefig(figure_name)
    plt.close()
    print("\n\nPLOT available in /content/VariationalBRAND/tests/figure.png")

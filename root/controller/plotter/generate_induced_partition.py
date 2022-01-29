import matplotlib
from root.controller.plotter.plot_covariance_ellipses import  plot_cov_ellipse
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp


def generate_induced_partition(Y,labels_pred, robust_mean, hyperparameters_model: HyperparametersModel, variational_parameters: VariationalParameters, cov_ellipse):
    #print(robust_mean)

    print('Clusters\' numerosity')
    #print(labels_pred)
    unique_clusters = np.unique(np.array(labels_pred))
    num_clusters = len(unique_clusters)

    for i in unique_clusters:
        print('cluster ', i, ': ',labels_pred.count(i))

    #plt.scatter(Y[:, 0], Y[:, 1], c=[matplotlib.cm.get_cmap("Spectral")(float(i) / num_clusters) for i in labels_pred])

    if (hyperparameters_model.p == 2):
        plt.scatter(Y[:, 0], Y[:, 1], c=[matplotlib.cm.get_cmap("Spectral")(float(i) / num_clusters) for i in labels_pred])

        for i in range(hyperparameters_model.J):
            plt.scatter(robust_mean[i][0], robust_mean[i][1], color='red')
        for i in unique_clusters:
            plt.scatter(variational_parameters.nIW.mu[i, 0], variational_parameters.nIW.mu[i, 1], color='black')

            #print(variational_parameters.nIW.phi[i],variational_parameters.nIW.mu[i, :] )
            if cov_ellipse:
                plot_cov_ellipse(variational_parameters.nIW.mu[i],
                                 variational_parameters.nIW.phi[i]/(variational_parameters.nIW.nu[i] - hyperparameters_model.p -1))

        p = hyperparameters_model.p
        M = hyperparameters_model.M
        figure_name = 'figure_' + str(p) + '_' + str(M)
        figure_filetype = '.png'
        if(cov_ellipse):
            figure_name = figure_name + '-ellipses' + figure_filetype
        else:
            figure_name = figure_name + figure_filetype

        plt.savefig(figure_name)
        plt.close()
        print("\n\nPLOT available in /content/VariationalBRAND/tests/figure.png")


def generate_induced_partition_iter(Y, robust_mean, iter, hyperparameters_model: HyperparametersModel, variational_parameters: VariationalParameters, cov_ellipse):
    ll = []
    for i in range(Y.shape[0]):
        ll.append(jnp.argmax(variational_parameters.phi_m_k[i, :]))


#    print('Clusters\' numerosity')
#    print(ll)
    unique_clusters = np.unique(np.array(ll))
    num_clusters = len(unique_clusters)

#    for i in unique_clusters:
#        print('cluster ', i, ': ',ll.count(i))
#        print(robust_mean)


    if (hyperparameters_model.p == 2):
        plt.scatter(Y[:, 0], Y[:, 1], c=[matplotlib.cm.get_cmap("Spectral")(float(i) / num_clusters) for i in ll])

        for i in range(hyperparameters_model.J):
            plt.scatter(robust_mean[i][0], robust_mean[i][1], color='red')
        for i in unique_clusters:
            plt.scatter(variational_parameters.nIW.mu[i, 0], variational_parameters.nIW.mu[i, 1], color='black')
        #    print(variational_parameters.nIW.phi[i],variational_parameters.nIW.mu[i, :] )
            if cov_ellipse:
                plot_cov_ellipse(variational_parameters.nIW.mu[i],
                                 variational_parameters.nIW.phi[i]/(variational_parameters.nIW.nu[i] - hyperparameters_model.p -1))

        p = hyperparameters_model.p
        M = hyperparameters_model.M
        figure_name = str(iter) + str(p) + '_' + str(M)

        figure_filetype = '.png'
        if(cov_ellipse):
            figure_name = figure_name + '-ellipses' + figure_filetype
        else:
            figure_name = figure_name + figure_filetype

        plt.savefig(figure_name)
        plt.close()
        print("\n\nPLOT available in /content/VariationalBRAND/tests/figure.png")

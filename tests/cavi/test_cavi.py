import random
from unittest import TestCase

import decorator
import matplotlib

from root.controller.cavi.cavi import cavi
from root.controller.cavi.init_cavi.init_cavi import init_cavi
from root.controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.sample_data_handler.utils import get_training_set_example
from root.controller.specify_user_input.specify_user_input import specify_user_input
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters
from jax import numpy as jnp
import numpy as np

def generate_induced_partition(Y, robust_mean, variational_parameters: VariationalParameters):
    import matplotlib.pyplot as plt
    from jax import numpy as jnp
    ll = []
    for i in range(750):
        ll.append(jnp.argmax(variational_parameters.phi_m_k[i, :]))


    print('Clusters\' numerosity')
    unique_clusters = np.unique(np.array(ll))

    for i in unique_clusters:
        print('cluster ', i, ': ',ll.count(i))

    plt.scatter(Y[:, 0], Y[:, 1], c=[matplotlib.cm.get_cmap("Spectral")(float(i) / 5) for i in ll])


    print(robust_mean)
    for i in range(3):
        plt.scatter(robust_mean[i][0], robust_mean[i][1], color='red')
    for i in range(13):
        plt.scatter(variational_parameters.nIW.mu[i, 0], variational_parameters.nIW.mu[i, 1], color='black')
    # plt.show()
    plt.savefig('figure.png')
    print("\n\nPLOT available in /content/VariationalBRAND/tests/figure.png")

class Test(TestCase):
    def test_cavi(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        # variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        # cavi(Y, hyperparameters_model, user_input_parameters)
        # print(variational_parameters.nIW_DP_VAR.mu)
        # print(variational_parameters.nIW_MIX_VAR.mu)

        variational_parameters = cavi(Y, hyperparameters_model, user_input_parameters)

        generate_induced_partition(Y, list_robust_mean, variational_parameters)
import random
from unittest import TestCase

import decorator
import matplotlib
import numpy
from numpy import tile

from controller.plotter.generate_induced_partition import generate_induced_partition
from controller.sample_data_handler.data_generator import generate_some_data_example
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

class Test(TestCase):
    def test_cavi(self):
        from numpy import loadtxt
        #data = loadtxt('data.csv', delimiter=',')
        data = loadtxt('Data_Luca.csv', delimiter=',')
        #data = generate_some_data_example()
        Y = data

        # num_clusters= 5
        # num_classes_training = 3
        #Y_training, num_classes_training = get_training_set_example(Y, num_clusters, num_classes_training)

        num_classes_training = 2
        Y_training = numpy.vstack([Y[0:299,:], Y[600:899,:]])

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat, Y)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)
        # variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        # cavi(Y, hyperparameters_model, user_input_parameters)
        # print(variational_parameters.nIW_DP_VAR.mu)
        # print(variational_parameters.nIW_MIX_VAR.mu)

        variational_parameters = cavi(Y, hyperparameters_model, user_input_parameters)

        generate_induced_partition(Y, list_robust_mean,hyperparameters_model, variational_parameters)
from unittest import TestCase
from root.controller.cavi.init_cavi.init_cavi import init_cavi
from root.controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.sample_data_handler.utils import get_training_set_example
from root.controller.specify_user_input.specify_user_input import specify_user_input
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters
from jax import numpy as jnp


class Test(TestCase):
    def test_init_cavi(self):
        from numpy import loadtxt
        data = loadtxt('data.csv', delimiter=',')
        Y = data

        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters)
        variational_parameters : VariationalParameters = init_cavi(user_input_parameters)

        #check conversion to jnp array
        self.assertEqual(type(variational_parameters.phi_m_k), type(jnp.array([])))

        self.assertEqual(variational_parameters.phi_m_k.shape, (500, 8))
from unittest import TestCase

from controller.cavi.init_cavi.init_cavi import init_cavi
from controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from controller.sample_data_handler.data_generator import generate_some_data_example
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from controller.sample_data_handler.utils import get_training_set_example
from controller.specify_user_input.user_input import specify_user_input
from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters
from jax import numpy as jnp


class Test(TestCase):
    def test_init_cavi(self):
        Y = generate_some_data_example()
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters)
        variational_parameters : VariationalParameters = init_cavi(user_input_parameters)

        #check conversion to jnp array
        self.assertEqual(type(variational_parameters.phi_m_k), type(jnp.array([])))

        self.assertEqual(variational_parameters.phi_m_k.shape, (500, 8))
from unittest import TestCase
from jax import numpy as jnp
from controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from controller.sample_data_handler.data_generator import generate_some_data_example
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from controller.sample_data_handler.utils import get_training_set_example
from controller.specify_user_input.user_input import specify_user_input
from model.hyperparameters_model import HyperparametersModel

class Test(TestCase):
    def test_set_hyperparameters(self):
        Y = generate_some_data_example()
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)

        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

        print(hyperparameters_model.nIW_DP_0.mu[0].shape)

        self.assertEqual(hyperparameters_model.nIW_DP_0.mu[0].shape, user_input_parameters.mu_0_DP[0].shape)
        self.assertEqual(len(hyperparameters_model.nIW_DP_0.mu), len(user_input_parameters.mu_0_DP))
        self.assertEqual(hyperparameters_model.nIW_DP_0.nu[0].shape, user_input_parameters.nu_0_DP[0].shape)
        self.assertEqual(len(hyperparameters_model.nIW_DP_0.nu), len(user_input_parameters.nu_0_DP))

        self.assertEqual(hyperparameters_model.nIW_DP_0.lambdA[0].shape, user_input_parameters.lambda_0_DP[0].shape)
        self.assertEqual(len(hyperparameters_model.nIW_DP_0.lambdA), len(user_input_parameters.lambda_0_DP))
        self.assertEqual(hyperparameters_model.nIW_DP_0.phi[0].shape, user_input_parameters.PHI_0_DP[0].shape)
        self.assertEqual(len(hyperparameters_model.nIW_DP_0.phi), len(user_input_parameters.PHI_0_DP))

        self.assertEqual(type(hyperparameters_model.nIW_DP_0.mu), type(jnp.array([])))


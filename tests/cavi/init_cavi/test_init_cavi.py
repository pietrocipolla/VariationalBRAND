from unittest import TestCase
from jax import numpy as jnp
from controller.cavi.init_cavi.init_cavi import init_cavi
from controller.sample_data_handler.data_generator import generate_some_data_example
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from controller.sample_data_handler.utils import get_training_set_example
from controller.specify_user_input.specify_user_input import specify_user_input
from model.variational_parameters import VariationalParameters


class Test(TestCase):
    def test_init_cavi(self):
        Y = generate_some_data_example()
        Y_training, num_classes_training = get_training_set_example(Y)

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat)
        variational_parameters : VariationalParameters = init_cavi(user_input_parameters)

        #check conversion to jnp array
        self.assertEqual(type(variational_parameters.phi_m_k), type(jnp.array([])))

        #print(user_input_parameters.eta_k)

        #print(variational_parameters.eta_k)

        self.assertEqual(variational_parameters.phi_m_k.shape, (500, 8))
        self.assertEqual(len(variational_parameters.eta_k), 4)

        self.assertEqual(len(variational_parameters.a_k_beta), 4)
        self.assertEqual(len(variational_parameters.b_k_beta), 4)

        # NIW_DP_VAR
        self.assertEqual(len(variational_parameters.nIW_DP_VAR.mu), 5)
        self.assertEqual(variational_parameters.nIW_DP_VAR.mu[0].shape, (2,))

        self.assertEqual(len(variational_parameters.nIW_DP_VAR.nu), 5)

        self.assertEqual(len(variational_parameters.nIW_DP_VAR.lambdA), 5)

        self.assertEqual(len(variational_parameters.nIW_DP_VAR.phi), 5)
        self.assertEqual(variational_parameters.nIW_DP_VAR.phi[0].shape, (2, 2))

        # NIW_MIX_VAR
        self.assertEqual(variational_parameters.nIW_MIX_VAR.mu[0].shape, (2,))
        self.assertEqual(variational_parameters.nIW_MIX_VAR.nu.shape, (3,))
        self.assertEqual(len(variational_parameters.nIW_MIX_VAR.lambdA), 3)
        self.assertEqual(len(variational_parameters.nIW_MIX_VAR.phi), 3)
        self.assertEqual(variational_parameters.nIW_MIX_VAR.phi[0].shape, (2, 2))
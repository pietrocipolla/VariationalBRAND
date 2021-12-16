from unittest import TestCase

from controller.cavi.cavi import cavi
from controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from controller.partition_inducer.partition_inducer import generate_induced_partition
from controller.sample_data_handler.data_generator import generate_some_data_example
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from controller.sample_data_handler.utils import get_training_set_example
from controller.specify_user_input.specify_user_input import specify_user_input
from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters


class Test(TestCase):
    def test_generate_induced_partition(self):
        Y = generate_some_data_example()

        # modify and pick a subset of Y for calculating robust parameters on Y_training
        Y_training, num_classes_training = get_training_set_example(Y)

        # automatic robust parameters from y_training and num of training classes
        robust_mean, robust_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

        # STEP 2
        # modify specify_user_input to change hyperparameters_setter to match your data
        user_input_parameters = specify_user_input(robust_mean, robust_inv_cov_mat)

        # the rest of the code is automatic
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

        # CAVI (init + update + elbo)
        variational_parameters: VariationalParameters = cavi(Y, hyperparameters_model, user_input_parameters)

        # INDUCED PARTITION
        generate_induced_partition(Y, robust_mean, variational_parameters)

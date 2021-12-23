from unittest import TestCase
import numpy
from controller.cavi.init_cavi.init_cavi import init_cavi
from controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from controller.specify_user_input.specify_user_input import specify_user_input
from controller.time_tracker.clones.elbo_calculator_time_tracker import elbo_calculator_time_tracker
from controller.time_tracker.time_tracker import TimeTracker
from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters


class Test(TestCase):
    def test_elbo_calculator_time_tracker(self):
        from numpy import loadtxt
        # data = loadtxt('data.csv', delimiter=',')
        data = loadtxt('Data_Luca.csv', delimiter=',')
        # data = generate_some_data_example()
        Y = data

        # num_clusters = 5
        # num_classes_training = 3
        # Y_training, num_classes_training = get_training_set_example(Y, num_clusters, num_classes_training)

        num_classes_training = 2
        Y_training = numpy.vstack([Y[0:299, :], Y[600:899, :]])

        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat, Y)

        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

        variational_parameters: VariationalParameters = init_cavi(user_input_parameters)

        elbo_calculator_time_tracker(Y, hyperparameters_model, variational_parameters, hyperparameters_model.p)

        # Performance
        TimeTracker.print_performance()
        TimeTracker.plot_elbo_calculator_performance()

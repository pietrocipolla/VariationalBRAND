import random
from unittest import TestCase


import matplotlib
import numpy
from numpy import tile

from controller.time_tracker.time_tracker import TimeTracker
from root.controller.plotter.generate_elbo_plot import generate_elbo_plot
from root.controller.plotter.generate_induced_partition import generate_induced_partition
from root.controller.sample_data_handler.data_generator import generate_some_data_example
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
from numpy import loadtxt

class Test(TestCase):
    def test_cavi(self):
        # STEP 1
        # modify to generate_some_data or load your own data
        # data = generate_some_data_example()
        # data = loadtxt('data.csv', delimiter=',')

        tic = TimeTracker.start()
        data = loadtxt('Y_testing.csv', delimiter=',')
        Y = data

        # modify and pick a subset of Y for calculating robust parameters on Y_training
        # data.csv
        # num_clusters= 5
        # num_classes_training = 3
        # Y_training, num_classes_training = get_training_set_example(Y, num_clusters, num_classes_training)

        # luca
        # num_classes_training = 2
        # Y_training = numpy.vstack([Y[0:299, :], Y[600:899, :]])

        # not_small_brand
        # num_classes_training = 3
        # Y_training = numpy.vstack([Y[0:199, :], Y[200:499, :],Y[500:749, :]])

        # seed_data y training
        num_classes_training = 2
        Y_training = loadtxt('X_training.csv', delimiter=',')

        # seed dataset robust parameters
        labels = loadtxt('labels.csv', delimiter=',')
        list_robust_mean, list_inv_cov_mat = calculate_robust_parameters_labels(Y_training, num_classes_training,
                                                                                labels)
        print('list_robust_mean', list_robust_mean)

        # automatic robust parameters from y_training and num of training classes
        # list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)
        # print('list_robust_mean' ,list_robust_mean)

        # STEP 2
        # modify specify_user_input to change hyperparameters_setter to match your data
        user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat, Y)

        # !the rest of the code is automatic

        # automatic set of hyperparameters from previously specified user input
        hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

        # CAVI (init + update + elbo)
        variational_parameters, elbo_values = cavi(Y, list_robust_mean, hyperparameters_model, user_input_parameters)

        TimeTracker.stop_and_save('main', tic)
        TimeTracker.plot_main_performance()
        TimeTracker.print_performance()
        # Generate figure of induced partition
        generate_induced_partition(Y, list_robust_mean, hyperparameters_model, variational_parameters,
                                   cov_ellipse=False)
        generate_induced_partition(Y, list_robust_mean, hyperparameters_model, variational_parameters, cov_ellipse=True)

        # Plot elbo
        generate_elbo_plot(elbo_values)
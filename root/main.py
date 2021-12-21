import numpy
from controller.plotter.generate_induced_partition import generate_induced_partition
from controller.plotter.generate_elbo_plot import generate_elbo_plot
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.cavi.cavi import cavi
from root.controller.specify_user_input.specify_user_input import specify_user_input
from root.controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from root.model.hyperparameters_model import HyperparametersModel
from numpy import loadtxt
from root.controller.sample_data_handler.data_generator import generate_some_data_example
from root.controller.sample_data_handler.utils import get_training_set_example

if __name__ == '__main__':
    #STEP 1
    #modify to generate_some_data or load your own data
    #data = generate_some_data_example()
    # data = loadtxt('data.csv', delimiter=',')
    data = loadtxt('Data_Luca.csv', delimiter=',')
    Y = data

    #modify and pick a subset of Y for calculating robust parameters on Y_training
    # num_clusters= 5
    # num_classes_training = 3
    #Y_training, num_classes_training = get_training_set_example(Y, num_clusters, num_classes_training)
    num_classes_training = 2
    Y_training = numpy.vstack([Y[0:299, :], Y[600:899, :]])

    #automatic robust parameters from y_training and num of training classes
    list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

    # STEP 2
    #modify specify_user_input to change hyperparameters_setter to match your data
    user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat, Y)

    #!the rest of the code is automatic

    #automatic set of hyperparameters from previously specified user input
    hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

    #CAVI (init + update + elbo)
    variational_parameters , elbo_values = cavi(Y, hyperparameters_model, user_input_parameters)

    #Generate figure of induced partition
    generate_induced_partition(Y, list_robust_mean, hyperparameters_model, variational_parameters)

    #Plot elbo
    generate_elbo_plot(elbo_values)


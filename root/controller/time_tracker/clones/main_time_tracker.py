from root.controller.sample_data_handler.utils import get_training_set_example
from root.controller.time_tracker.clones.cavi_time_tracker import cavi_time_tracker
from root.controller.time_tracker.time_tracker import TimeTracker
from root.controller.plotter.generate_induced_partition import generate_induced_partition
from root.controller.plotter.generate_elbo_plot import generate_elbo_plot
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from root.model.hyperparameters_model import HyperparametersModel
from numpy import loadtxt

def main_time_tracker():
    #STEP 1
    tic = TimeTracker.start()

    #modify to generate_some_data or load your own data
    #data = generate_some_data_example()
    data = loadtxt('data.csv', delimiter=',')
    #data = loadtxt('Data_Luca.csv', delimiter=',')
    Y = data

    TimeTracker.stop_and_save('load_data', tic)


    #modify and pick a subset of Y for calculating robust parameters on Y_training
    tic = TimeTracker.start()

    num_clusters= 5
    num_classes_training = 3
    Y_training, num_classes_training = get_training_set_example(Y, num_clusters, num_classes_training)
    # num_classes_training = 2
    # Y_training = numpy.vstack([Y[0:299, :], Y[600:899, :]])

    TimeTracker.stop_and_save('get_training_set', tic)

    #automatic robust parameters from y_training and num of training classes
    tic = TimeTracker.start()

    list_robust_mean, list_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

    TimeTracker.stop_and_save('calculate_robust_parameters', tic)

    # STEP 2
    #modify specify_user_input to change hyperparameters_setter to match your data
    tic = TimeTracker.start()

    user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat, Y)

    TimeTracker.stop_and_save('specify_user_input', tic)

    #!the rest of the code is automatic

    #automatic set of hyperparameters from previously specified user input
    tic = TimeTracker.start()

    hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

    TimeTracker.stop_and_save('set_hyperparameters', tic)

    #CAVI (init + update + elbo)
    tic = TimeTracker.start()

    variational_parameters, elbo_values = cavi_time_tracker(Y, list_robust_mean, hyperparameters_model, user_input_parameters)

    TimeTracker.stop_and_save('cavi', tic)

    #Generate figure of induced partition
    tic = TimeTracker.start()

    generate_induced_partition(Y, list_robust_mean, hyperparameters_model, variational_parameters, cov_ellipse=False)
    generate_induced_partition(Y, list_robust_mean, hyperparameters_model, variational_parameters, cov_ellipse=True)

    TimeTracker.stop_and_save('generate_induced_partition', tic)

    #Plot elbo
    tic = TimeTracker.start()

    generate_elbo_plot(elbo_values)

    TimeTracker.stop_and_save('generate_elbo_plot', tic)

    #Performance
    TimeTracker.get_performance()
    TimeTracker.plot_main_performance()
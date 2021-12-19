from root.controller.sample_data_handler.data_generator import generate_some_data_example
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from root.controller.cavi.cavi import cavi
from root.controller.sample_data_handler.utils import get_training_set_example
from root.controller.specify_user_input.specify_user_input import specify_user_input
from root.controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from root.model.hyperparameters_model import HyperparametersModel
from numpy import loadtxt

if __name__ == '__main__':
    #STEP 1
    #modify generate_some_data or load youw own data
    Y = generate_some_data_example()
    #Y = loadtxt('data.csv', delimiter=',')

    #modify and pick a subset of Y for calculating robust parameters on Y_training
    num_clusters = 5
    num_classes_learning = 3
    Y_training, num_classes_training = get_training_set_example(Y, num_clusters, num_classes_learning)

    #automatic robust parameters from y_training and num of training classes
    robust_mean, robust_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

    # STEP 2
    #modify specify_user_input to change hyperparameters_setter to match your data
    user_input_parameters = specify_user_input(robust_mean, robust_inv_cov_mat, Y)

    #the rest of the code is automatic
    hyperparameters_model : HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

    # CAVI (init + update + elbo)
    variational_parameters = cavi(Y, hyperparameters_model, user_input_parameters)

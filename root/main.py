from controller.sample_data_handler.data_generator import generate_some_data
from controller.sample_data_handler.robust_calculator import calculate_robust_parameters
from controller.cavi.cavi import cavi
from controller.sample_data_handler.utils import get_training_set
from controller.user_input import specify_user_input
from controller.hyperparameters.set_hyperparameters import set_hyperparameters

if __name__ == '__main__':
    #STEP 1
    #modify generate_some_data to create your own data
    Y = generate_some_data()

    #modify and pick a subset of Y for calculating robust parameters on Y_training
    Y_training, num_classes_training = get_training_set(Y)

    #automatic robust parameters from y_training and num of training classes
    robust_mean, robust_inv_cov_mat = calculate_robust_parameters(Y_training, num_classes_training)

    # STEP 2
    #modify specify_user_input to change hyperparameters to match your data
    user_input_parameters = specify_user_input(robust_mean, robust_inv_cov_mat)

    #the rest of the code is automatic
    hyperparameters_model = set_hyperparameters(user_input_parameters)

    # CAVI (init + update + elbo)
    n_iter = 1000
    variational_parameters, elbo_values = cavi(Y, hyperparameters_model,user_input_parameters, n_iter)

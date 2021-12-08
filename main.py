from controller.cavi.cavi import cavi
from controller.hyperparameters.set_hyperparameters_fake import set_hyperparameters_fake
from controller.model_parameters_calculator.minimum_covariance_determinant import calculate_robust_parameters
from controller.sample_data_hanlder.data_handler import generate_learning_and_test_sets
from model.hyperparameters_model.hyperparameters_model import HyperparametersModel

if __name__ == '__main__':
    #GENERATE SAMPLE DATA
    X_learning, labels_learning, num_classes_learning, \
    X_test, labels_test, num_classes_test,\
    num_classes, n_samples, X\
          = generate_learning_and_test_sets()

    # ROBUST PARAMETERS
    #todo creare funzione che chiede due hyper robust che non escono dal calculate
    #piu settare gia la struttura
    robust_covariance, robust_mean = calculate_robust_parameters(X_learning, labels_learning, num_classes_learning)
    #print(robust_covariance, robust_mean)
    #ritorna vettore di covaraince e mean per ogni cluster training set


    # HYPERPARAMETERS
    ask_hyperparameters_from_user_input = False

    hyperparameters_model : HyperparametersModel = \
        set_hyperparameters_fake(ask_hyperparameters_from_user_input, num_classes,
                                 num_classes_learning, num_classes_test, robust_mean, n_samples)

    # CAVI (init + update + elbo)
    n_iter = 1000
    #data all data
    data = X
    variational_parameters, elbo_values = cavi(data, hyperparameters_model, n_iter)

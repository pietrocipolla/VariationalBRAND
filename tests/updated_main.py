from controller.specify_user_input.specify_user_input import specify_user_input
from controller.time_tracker.time_tracker import TimeTracker
from root.controller.plotter.generate_induced_partition import generate_induced_partition
from root.controller.plotter.generate_elbo_plot import generate_elbo_plot
from root.controller.sample_data_handler.robust_calculator import calculate_robust_parameters_labels
from root.controller.cavi.cavi import cavi
from root.controller.hyperparameters_setter.set_hyperparameters import set_hyperparameters
from root.model.hyperparameters_model import HyperparametersModel
from numpy import loadtxt
import copy
import numpy as np
from controller.sample_data_handler.test_init_kmeans import test_mu_var_DP_init_kmeans
from root.model.user_input_model import UserInputModel
from unittest import TestCase


def specify_user_input(robust_mean, robust_inv_cov_mat, Y, NUM_CLASSES_TRAINING):
    # Numero di componenti
    p = Y.shape[1]
    # Numero di classi nel training set
    J = NUM_CLASSES_TRAINING
    # Numero di classi massime nel Dirichlet Process
    #T = 20
    T = 10

    # Num iteration and tolerance cavi
    n_iter = 1000
    tol = 1e-4

    #HYPERPARAMETERS
    gamma = 5
    # gamma -> parametro dello Stick Breaking -> scalare
    # iperparametro tra 1 e 50 tipo oppure buttarci su una distribuzione e una prior

    #a_dir_k = np.ones(J + 1)*2
    a_dir_k = np.ones(J + 1) * 0.1

    # a_dir_k -> vettore delle componenti della Dir0ichlet -> vettore di (J+1) componenti
    # J = num_classes_learning

    #nIW_DP_0
    # 	mu_0_DP -> vettore (p) componenti
    # 	nu_0_DP -> scalare
    # 	lambda_0_DP -> scalare
    # 	PHI_0_DP -> Matrice (pxp)

    #mu_0_DP = np.ones(p)  # così che sia comunque della forma n_elems x p
    mu_0_DP = np.zeros(p)

    nu_0_DP = np.array([10])

    lambda_0_DP = np.array([0.03])


    PHI_0_DP = np.multiply(np.identity(p), 10)


    #NIW_MIX_0
    # mu_0_MIX -> matrice (Jxp)
    #           -> riga per riga ci sono le medie delle componenti della mistura
    #               (mu_0_MIX[i,:] -> media della (i+1)-esima NIW della mistura)
    #
    # nu_0_MIX -> vettore (J) componenti
    #               (nu_0_MIX[i] = nu_0 della (i+1) esima NIW della mistura)
    #
    # lambda_0_MIX -> vettore (J) componenti
    #               (lambda_0_MIX[i] = lambda_0 della (i+1)  esima NIW della mistura)
    #
    # PHI_0_MIX -> vettore di matrici (Jxpxp), sostanzialmente un ndarray
    #           -> PHI_0_MIX[i,:,:] = Matrice PHI della (i+1)esima NIW della mistura

    mu_0_MIX = robust_mean

    # nu_0_MIX = np.multiply(np.ones(J), 5.4)
    #
    # lambda_0_MIX = np.ones(J)*100

    # nu_0_MIX = np.multiply(np.ones(J), 5)
    #
    # lambda_0_MIX = np.ones(J)

    nu_0_MIX = np.multiply(np.ones(J), 10)

    lambda_0_MIX = np.ones(J)


    PHI_0_MIX = robust_inv_cov_mat # TODO trovare inizializzazione più furba di 0_MIX

    #VARIATIONAL PARAMETERS
    M = Y.shape[0]
    phi_m_k_temp = np.zeros((M, J + T))

    # for k in range(J):
    #         phi_m_k_temp[:, k] = 1 / (J + 1)
    # for k in range(J, J + T):
    #         phi_m_k_temp[:, k] = (1 / (J + 1)) * (0.5 ** (k - J + 1)) * (1 / (1 - 0.5 ** T))

    Phi_m_k = phi_m_k_temp
    # > parametri delle multinomiali -> matrice (Mx(J+T))
    # -> phi_m_k[i:] = parametri della (i+1)esima multinomiale
    # -> phi_m_k[i,j] = prob che y_i sia nel cluster j

    eta_k = copy.deepcopy(a_dir_k)

    a_k_beta = np.ones(T - 1)

    b_k_beta = np.multiply(np.ones(T - 1), gamma)

    # NIW_DP_VAR
    # mu_var_DP -> matrice (Txp)
    # -> riga per riga ci sono le medie delle componenti della misturaDP
    #    (mu_var_DP[i,:] -> media della (i+1)-esima NIW della misturaDP)

    # nu_var_DP -> vettore (T) componenti
    # (nu_var_MIX[i] = nu_var della (i+1) esima NIW della misturaDP)

    # lambda_var_DP -> vettore (T) componenti
    # (lambda_var_DP[i] = lambda_var della (i+1) esima NIW della misturaDP)

    # PHI_var_DP -> vettore di matrici (Txpxp), sostanzialmente un ndarray
    # -> PHI_var_DP[i,:,:] = Matrice PHI della (i+1)esima NIW della misturaDP

    #mu_var_DP = np.repeat(mu_0_DP, repeats=T, axis=0) #todo old chec se era sbagliato visto che vogliamo matrice Txp


    #mu_var_DP = np.tile(mu_0_DP, (T,1))
    # print(np.tile(mu_0_DP, (T,1)))
    #mu_var_DP = test_mu_var_DP_init()
    mu_var_DP = test_mu_var_DP_init_kmeans(Y, T)

    # print('mu_0_DP', mu_0_DP)
    # print('mu_var_DP', mu_var_DP)

    nu_var_DP = np.multiply(np.ones(T), nu_0_DP)
    nu_var_DP = np.reshape(nu_var_DP, T)
    # print('nu_var_DP', nu_var_DP)
    # print('nu_0_DP', nu_0_DP)

    lambda_var_DP = np.multiply(np.ones(T), lambda_0_DP)
    lambda_var_DP = np.reshape(lambda_var_DP, T)
    #print('lambda_var_DP', lambda_var_DP)

    PHI_var_DP = np.repeat(copy.deepcopy(PHI_0_DP)[None,:], repeats = T, axis = 0)
    PHI_var_DP = np.reshape(PHI_var_DP, (T, p, p))
    #print('PHI_var_DP', PHI_var_DP)

    # print('PHI_var_DP')
    # print(PHI_var_DP)

    #NIW_MIX_VAR:
    # mu_VAR_MIX -> matrice (Jxp)
    #           -> riga per riga ci sono le medie delle componenti della mistura
    #               (mu_VAR_MIX[i,:] -> media della (i+1)-esima NIW della mistura)
    #
    # nu_VAR_MIX -> vettore (J) componenti
    #               (nu_VAR_MIX[i] = nu_0 della (i+1) esima NIW della mistura)
    #
    # lambda_VAR_MIX -> vettore (J) componenti
    #               (lambda_VAR_MIX[i] = lambda_0 della (i+1)  esima NIW della mistura)
    #
    # PHI_VAR_MIX -> vettore di matrici (Jxpxp), sostanzialmente un ndarray
    #           -> PHI_VAR_MIX[i,:,:] = Matrice PHI della (i+1)esima NIW della mistura

    mu_VAR_MIX = copy.deepcopy(mu_0_MIX)
    #print(mu_VAR_MIX)

    nu_VAR_MIX = copy.deepcopy(nu_0_MIX)
    #print('nu_VAR_MIX', nu_VAR_MIX)

    lambda_VAR_MIX = copy.deepcopy(lambda_0_MIX)

    PHI_VAR_MIX = copy.deepcopy(PHI_0_MIX)

    return UserInputModel(
        J = J,
        T = T,

        n_iter = n_iter,
        tol = tol,

        gamma = gamma,
        a_dir_k = a_dir_k,

        mu_0_DP = mu_0_DP,
        nu_0_DP = nu_0_DP,
        lambda_0_DP = lambda_0_DP,
        PHI_0_DP = PHI_0_DP,

        mu_0_MIX = mu_0_MIX,
        nu_0_MIX = nu_0_MIX,
        lambda_0_MIX = lambda_0_MIX,
        PHI_0_MIX = PHI_0_MIX,

        Phi_m_k = Phi_m_k,
        eta_k = eta_k,
        a_k_beta = a_k_beta,
        b_k_beta = b_k_beta,

        mu_var_DP = mu_var_DP,
        nu_var_DP = nu_var_DP,
        lambda_var_DP = lambda_var_DP,
        PHI_var_DP = PHI_var_DP,

        mu_VAR_MIX = mu_VAR_MIX,
        nu_VAR_MIX = nu_VAR_MIX,
        lambda_VAR_MIX = lambda_VAR_MIX,
        PHI_VAR_MIX = PHI_VAR_MIX,
    )

def updated_main():
    # SPECIFY DATASETS' INFO
    Y_ALL_FILENAME = 'Y.csv'
    Y_TRAINING_FILENAME = 'Y_training.csv'
    LABELS_TRAINING_FILENAME = 'labels_training.csv'
    NUM_CLASSES_TRAINING = 3

    tic = TimeTracker.start()

    #STEP 1
    #load training dataset and labels
    Y_training = loadtxt(Y_TRAINING_FILENAME, delimiter=',')
    labels_training = loadtxt(LABELS_TRAINING_FILENAME, delimiter=',')

    # calculate robust parameters
    list_robust_mean, list_inv_cov_mat = calculate_robust_parameters_labels(Y_training, NUM_CLASSES_TRAINING, labels_training)

    # STEP 2
    # load entire dataset
    Y = loadtxt(Y_ALL_FILENAME, delimiter=',')

    # specificy hyperparameters (! modify to match your hyperparameters)
    user_input_parameters = specify_user_input(list_robust_mean, list_inv_cov_mat, Y, NUM_CLASSES_TRAINING)

    # automatic set of hyperparameters given previous user input
    hyperparameters_model: HyperparametersModel = set_hyperparameters(user_input_parameters, Y)

    #CAVI (init + update + elbo)
    variational_parameters, elbo_values = cavi(Y, list_robust_mean, hyperparameters_model, user_input_parameters)

    TimeTracker.stop_and_save('main', tic)
    TimeTracker.plot_main_performance()
    TimeTracker.print_performance()

    #Generate induced partitions
    generate_induced_partition(Y, list_robust_mean, hyperparameters_model, variational_parameters, cov_ellipse=False)
    generate_induced_partition(Y, list_robust_mean, hyperparameters_model, variational_parameters, cov_ellipse=True)

    #Generate elbo plot
    generate_elbo_plot(elbo_values)


class Test(TestCase):
    def test_cavi(self):
        updated_main()
from model.variational_parameters.variational_parameters import VariationalParameters
from model.hyperparameters_model.hyperparameters_model import HyperparametersModel # non so se si faccia cosi
# from cavi.utils import useful_functions
from jax import numpy as jnp
import jax.scipy.special.digamma as jdigamma
import jax.scipy.special.gamma as jgamma

# QUI HO COPIATO E INCOLLATO



def update_parameters(data, hyperparameters, variational_parameters):
    #update parameters
    J = hyperparameters.J
    sum_phi_k = jnp.sum(variational_parameters.phi_m_k, axis=-1)

    mask = (sum_phi_k != 0)
    mask[0:J] = True
    sum_phi_k = sum_phi_k[mask]

    T_true = len(sum_phi_k) - J

    # Define function for each family update:
    # Dirichlet_eta
    update_dirichlet(variational_parameters, hyperparameters, sum_phi_k)
    # Beta_V
    update_beta(variational_parameters, hyperparameters, sum_phi_k)
    # NIW_mu_nu_lamda_Phi_mixture
    # NIW_mu_nu_lamda_Phi_DP
    # Multinomial_phi



    # Con una struttura di aggiornamento del genere conviene che i parametri vengano modificati con puntatori
    # per non copiare continuamente un oggetto sempre simile




    #update jax array
    return variational_parameters



# Scrivo qua in caso poi organizziamo meglio i file

############# UPDATE DIRICHLET ##############à
def update_dirichlet(variational_parameters, hyperparameters, sum_phi_k):
    J = hyperparameters.J
    temporary_phi_k = jnp.zeros(J+1)
    temporary_phi_k[0:-1] = sum_phi_k[:J]
    temporary_phi_k[-1] = jnp.sum(sum_phi_k[J:])

    variational_parameters.eta_k = hyperparameters.a_dir_k + temporary_phi_k


############# UPDATE BETA ##############
def update_beta(variational_parameters : VariationalParameters, hyperparameters : HyperparametersModel, sum_phi_k):
    J = hyperparameters.J

    remaining_probs = jnp.cumsum(jnp.flip(sum_phi_k[J:]))

    variational_parameters.a_k_beta = sum_phi_k[J:J+T] + 1
    variational_parameters.b_k_beta = hyperparameters.gamma + Tk * sum_phi_k[J:J+T]


###############################################################
###############          NIW _ MIX          ###################
###############################################################

############# UPDATE NIW ##############à
def update_NIW(y, variational_parameters, hyperparameters, sum_phi_k):
    J = hyperparameters.J
    phi_mk = variational_parameters.phi_m_k




    sum_y_phi = y.T @ phi_mk
    y_bar = eval_y_bar(sum_phi_k, sum_y_phi)

    update_NIW_MIX_mu(variational_parameters, hyperparameters, sum_y_phi, sum_phi_k)
    update_NIW_MIX_lambda(variational_parameters, hyperparameters, sum_phi_k)
    update_NIW_MIX_nu(variational_parameters, hyperparameters, sum_phi_k)
    update_NIW_MIX_PHI(variational_parameters, hyperparameters, sum_phi_k, y_bar, y, phi_mk)


def eval_y_bar(sum_phi_k, sum_y_phi):
    temp = jnp.diag(1/sum_phi_k)
    y_bar = sum_y_phi @ temp
    return y_bar

def update_NIW_MIX_mu(variational_parameters, hyperparameters, sum_y_phi, sum_phi_k):
    J = hyperparameters.J
    lambda0 = hyperparameters.nIW_MIX_0.lambda_0
    mu0 = hyperparameters.nIW_MIX_0.mu_0

    num = mu0.T @ lambda0 + sum_y_phi[:, :J]
    den = lambda0 + sum_phi_k[:J]
    mu_k = num/den

    variational_parameters.nIW_MIX_VAR.mu_0 = mu_k.T
    # return variational_parameters.nIW_MIX_VAR.mu_0_MIX

def update_NIW_MIX_lambda(variational_parameters,hyperparameters, sum_phi_k):
    J = hyperparameters.J
    variational_parameters.nIW_MIX_VAR.lambda_0_MIX = hyperparameters.nIW_MIX_0.lambda_0_MIX + sum_phi_k[:J]

def update_NIW_MIX_nu(variational_parameters,hyperparameters, sum_phi_k):
    J = hyperparameters.J
    variational_parameters.nIW_MIX_VAR.nu_0_MIX = hyperparameters.nIW_MIX_0.nu_0_MIX + sum_phi_k[:J]


def update_NIW_MIX_PHI(variational_parameters,hyperparameters, sum_phi_k, y_bar,y, phi_mk):

    PHI0 = hyperparameters.nIW_MIX_0.phi_0_MIX
    J = hyperparameters.J
    mu0 = hyperparameters.nIW_MIX_0.mu_0_MIX
    lambda0 = hyperparameters.nIW_MIX_0.lambda_0_MIX

    comp_2 = jnp.zeros(PHI0.shape)
    for k in range(J):
        curr_y_bar = y_bar[:, k]
        one_vec = jnp.ones((1, curr_y_bar.shape[0]))
        y_bar_matrix = (curr_y_bar @ one_vec)
        diff_y = y.T-y_bar_matrix
        phi_m = jnp.diag(phi_mk[:, k])
        comp_2[k, :, :] = diff_y @ phi_m @ diff_y.T # decvo salvarlo sotto forma di matrice

    comp_3 = jnp.zeros(PHI0.shape)
    for k in range(J):
        curr_y_bar = y_bar[:, k]
        diff = curr_y_bar - mu0[k,:].T
        diff_matrix = diff@diff.T
        coeff = lambda0[k]*sum_phi_k[k]/(lambda0[k]+sum_phi_k[k])
        comp_3[k, :, :] = coeff*diff_matrix

    variational_parameters.nIW_MIX_VAR.phi_0_MIX = PHI0 + comp_2 + comp_3


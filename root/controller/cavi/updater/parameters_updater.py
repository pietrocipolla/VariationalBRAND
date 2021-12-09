from model.variational_parameters import VariationalParameters
from model.hyperparameters_model import HyperparametersModel # non so se si faccia cosi
from model import NIW
from jax import numpy as jnp


def update_parameters(data, hyperparameters: HyperparametersModel, variational_parameters: VariationalParameters):
    #update parameters
    J = hyperparameters.J
    sum_phi_k = jnp.sum(variational_parameters.phi_m_k, axis=-1)

    mask = create_mask(sum_phi_k, J)
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


############# UTILS ##############
def create_mask(sum_phi_k,J):
    mask = (sum_phi_k != 0)
    mask[0:J] = True
    return mask

def eval_y_bar(sum_phi_k, sum_y_phi):
    temp = jnp.diag(1/sum_phi_k)
    y_bar = sum_y_phi @ temp            #(px(J+1t-true))*((J+T_true)x(J+T_true)) = px(J+T_true)
    return y_bar

# Scrivo qua in caso poi organizziamo meglio i file

############# UPDATE DIRICHLET ##############
def update_dirichlet(variational_parameters : VariationalParameters, hyperparameters, sum_phi_k):
    J = hyperparameters.J
    temporary_phi_k = jnp.zeros(J+1)
    temporary_phi_k[0:-1] = sum_phi_k[:J]
    temporary_phi_k[-1] = jnp.sum(sum_phi_k[J:])

    variational_parameters.eta_k = hyperparameters.a_dir_k + temporary_phi_k



############# UPDATE BETA ##############
def update_beta(variational_parameters : VariationalParameters, hyperparameters : HyperparametersModel, sum_phi_k):
    J = hyperparameters.J
    T = hyperparameters.T

    remaining_probs = jnp.cumsum(jnp.flip(sum_phi_k[J:]))
    remaining_probs = jnp.flip(remaining_probs)

    variational_parameters.a_k_beta = sum_phi_k[J:J+T] + 1
    variational_parameters.b_k_beta = hyperparameters.gamma + remaining_probs


###############################################################
###############          NIW _ MIX          ###################
###############################################################

############# UPDATE NIW ##############à
def update_NIW(y, variational_parameters : VariationalParameters, hyperparameters: HyperparametersModel, sum_phi_k, mask, T_true):
    phi_mk = variational_parameters.phi_m_k


    # supponendo y Mxp
    sum_y_phi = y.T @ phi_mk[mask]                          # (Mxp)T*(M*(J+T_true)) = px(J+T_true)
    y_bar = eval_y_bar(sum_phi_k, sum_y_phi)                # px(J+T_true)

    update_NIW_mu(variational_parameters, hyperparameters, sum_y_phi, sum_phi_k, T_true)
    update_NIW_lambda(variational_parameters, hyperparameters, sum_phi_k, T_true)
    update_NIW_nu(variational_parameters, hyperparameters, sum_phi_k)
    update_NIW_MIX_PHI(variational_parameters, hyperparameters, sum_phi_k, y_bar, y, phi_mk)


def update_NIW_mu(variational_parameters: VariationalParameters , hyperparameters_model: HyperparametersModel, sum_y_phi, sum_phi_k, T_true):
    # Estrazione parametri
    J = hyperparameters_model.J
    lambda0_DP = hyperparameters_model.nIW_DP_0.lambda_0_DP
    lambda0_MIX = hyperparameters_model.nIW_MIX_0.lambda_0_MIX
    mu0_DP = hyperparameters_model.nIW_DP_0.mu_0_DP.T           # pxT
    mu0_MIX = hyperparameters_model.nIW_MIX_0.mu_0_MIX.T        # pxJ

    lambda0 = jnp.concatenate((lambda0_MIX, lambda0_DP))          # J+T_true

    # MU
#    mu0_DP_vec = mu0_DP@one_vec                                 # pxT_true
    mu0 = jnp.concatenate((mu0_MIX, mu0_DP), axis=1)        # px(J+T_true)

    # CALCOLI AGGIORNAMENTO
    num = jnp.diag(lambda0) @ mu0 + jnp.ones((1, mu0.shape[0])) @ sum_y_phi      # px(J+T_true)
    den = lambda0 + sum_phi_k
    mu_k = num/den                                                          # px(J+T_true)

    variational_parameters.nIW_MIX_VAR.mu = mu_k[:J, :].T
    variational_parameters.nIW_DP_VAR.mu = mu_k[J:, :].T


def update_NIW_lambda(variational_parameters: VariationalParameters , hyperparameters_model: HyperparametersModel, sum_phi_k, T_true):
    J = hyperparameters_model.J
    lambda0_DP = hyperparameters_model.nIW_DP_0.lambda_0_DP
    lambda0_MIX = hyperparameters_model.nIW_MIX_0.lambda_0_MIX

#    one_vec = jnp.ones((1, T_true))

    # LAMBDA
#    lambda0_DP_vec = lambda0_DP * one_vec  # T_true
    lambda0 = jnp.concatenate((lambda0_MIX, lambda0_DP))  # J+T_true

# NON cambio ora i nomi ma forse meglio non chiamare con 0 i parametri variazionali (eg lambda_0_mix o mu_0_mix)
    variational_parameters.nIW_MIX_VAR.labmdA= (lambda0 + sum_phi_k)[:J]
    variational_parameters.nIW_DP_VAR.lambdA = (lambda0 + sum_phi_k)[J:]


def update_NIW_nu(variational_parameters: VariationalParameters,hyperparameters_model: HyperparametersModel, sum_phi_k, T_true):
    J = hyperparameters_model.J
    nu0_DP = hyperparameters_model.nIW_DP_0.nu_0_DP
    nu0_MIX = hyperparameters_model.nIW_MIX_0.nu_0_MIX

    #one_vec = jnp.ones((1, T_true))

    # NU
    #nu0_DP_vec = nu0_DP * one_vec  # T_true
    nu0 = jnp.concatenate((nu0_MIX, nu0_DP))  # J+T_true

    variational_parameters.nIW_MIX_VAR.nu = (nu0 + sum_phi_k)[:J]
    variational_parameters.nIW_DP_VAR.nu = (nu0 + sum_phi_k)[J:]

# modified untill here

def update_NIW_MIX_PHI(variational_parameters: VariationalParameters,hyperparameters: HyperparametersModel, sum_phi_k, y_bar,y, phi_mk, T_true):

    J = hyperparameters.J
    # si accede così?
    mu0_DP = hyperparameters.nIW_DP_0.mu.T              # px1
    mu0_MIX = hyperparameters.nIW_MIX_0.mu.T            # Jx1
    lambda0_DP = hyperparameters.nIW_DP_0.lambdA        # 1
    lambda0_MIX = hyperparameters.nIW_MIX_0.lambdA      # J
    PHI0_DP = hyperparameters.nIW_DP_0.phi              # 1xpxp
    PHI0_MIX = hyperparameters.nIW_MIX_0.phi            # Jxpxp

    y_bar_matrix = jnp.repeat(y_bar[:, jnp.newaxis, :], M, axis=1)
    mu0 = jnp.concatenate((mu0_MIX, mu0))  # J+T_true


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

    variational_parameters.nIW_MIX_VAR.phi = PHI0 + comp_2 + comp_3


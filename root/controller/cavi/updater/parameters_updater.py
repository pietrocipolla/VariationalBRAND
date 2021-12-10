from jax._src.numpy.linalg import jdet, jinv
from jax._src.scipy.special import jdigamma as digamma
from jax import numpy as jnp
from model.hyperparameters_model import HyperparametersModel
from model.variational_parameters import VariationalParameters

# Use jit and vectorization!

def update_parameters(data, hyperparameters: HyperparametersModel, variational_parameters: VariationalParameters):
    #update parameters
    J = hyperparameters.J
    sum_phi_k = jnp.sum(variational_parameters.phi_m_k, axis=0)

    #mask = create_mask(sum_phi_k, J)
    #sum_phi_k = sum_phi_k[mask]
    T_true = len(sum_phi_k) - J

    # Define function for each family update:
    # Dirichlet_eta
    update_dirichlet(variational_parameters, hyperparameters, sum_phi_k)
    # Beta_V
    update_beta(variational_parameters, hyperparameters, sum_phi_k)
    # NIW_mu_nu_lamda_Phi
    update_NIW(data, variational_parameters, hyperparameters, sum_phi_k, T_true)
    # Multinomial_phi
    update_phi_mk(data, variational_parameters, hyperparameters.T, hyperparameters.J)

    #update jax array
    return variational_parameters


############# UTILS ##############
def create_mask(sum_phi_k,J):
    mask = (sum_phi_k != 0)
    mask[0:J] = True
    return mask

def eval_y_bar(sum_phi_k, sum_y_phi):
    y_bar = sum_y_phi * 1/sum_phi_k            #(px(J+1t-true))*((J+T_true)x(J+T_true)) = px(J+T_true)
    return y_bar

# Scrivo qua in caso poi organizziamo meglio i file

############# UPDATE DIRICHLET ##############
def update_dirichlet(variational_parameters : VariationalParameters, hyperparameters, sum_phi_k):
    J = hyperparameters.J
    temporary_phi_k = jnp.zeros(J+1)
    temporary_phi_k = temporary_phi_k.at[0:-1].set(sum_phi_k[:J])
    temporary_phi_k = temporary_phi_k.at[-1].set(jnp.sum(sum_phi_k[J:]))

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
###############          NIW                ###################
###############################################################

############# UPDATE NIW ##############à
def update_NIW(y, variational_parameters : VariationalParameters, hyperparameters: HyperparametersModel, sum_phi_k, T_true):
    phi_mk = variational_parameters.phi_m_k

    # supponendo y Mxp
    sum_y_phi = y.T @ phi_mk                         # pxM*(M*(J+T_true)) = px(J+T_true)
    y_bar = eval_y_bar(sum_phi_k, sum_y_phi)                # px(J+T_true)

    update_NIW_mu(variational_parameters, hyperparameters, sum_y_phi, sum_phi_k, T_true)
    update_NIW_lambda(variational_parameters, hyperparameters, sum_phi_k, T_true)
    update_NIW_nu(variational_parameters, hyperparameters, sum_phi_k, T_true)
    update_NIW_PHI(variational_parameters, hyperparameters, sum_phi_k, y_bar, y, phi_mk, hyperparameters.T)


def update_NIW_mu(variational_parameters: VariationalParameters , hyperparameters_model: HyperparametersModel, sum_y_phi, sum_phi_k, T_true):
    # Estrazione parametri
    J = hyperparameters_model.J
    lambda0_DP = hyperparameters_model.nIW_DP_0.lambdA
    lambda0_MIX = hyperparameters_model.nIW_MIX_0.lambdA
    mu0_DP = hyperparameters_model.nIW_DP_0.mu.T           # pxT
    mu0_MIX = hyperparameters_model.nIW_MIX_0.mu.T        # pxJ

    one_vec = jnp.ones((1, T_true))

    # LAMBDA
    lambda0_MIX = jnp.reshape(lambda0_MIX, (1, len(lambda0_MIX)))
    lambda0_DP_vec = lambda0_DP * one_vec  # T_true
    lambda0 = jnp.concatenate((lambda0_MIX, lambda0_DP_vec), axis=1)          # J+T_true

    # MU
    mu0_DP_vec = jnp.repeat(mu0_DP[:, :], T_true, axis=1)                                 # pxT_true
    mu0 = jnp.concatenate((mu0_MIX, mu0_DP_vec), axis=1)        # px(J+T_tr

    # CALCOLI AGGIORNAMENTO
    num = lambda0 * mu0 + sum_y_phi      # px(J+T_true)
    den = lambda0 + sum_phi_k
    mu_k = num/den                                                          # px(J+T_true)

    variational_parameters.nIW_MIX_VAR.mu = mu_k[:, :J].T
    variational_parameters.nIW_DP_VAR.mu = mu_k[:, J:].T


def update_NIW_lambda(variational_parameters: VariationalParameters , hyperparameters_model: HyperparametersModel, sum_phi_k, T_true):
    J = hyperparameters_model.J
    lambda0_DP = hyperparameters_model.nIW_DP_0.lambdA
    lambda0_MIX = hyperparameters_model.nIW_MIX_0.lambdA

    one_vec = jnp.ones((1, T_true))

    # LAMBDA
    lambda0_MIX = jnp.reshape(lambda0_MIX, (1, len(lambda0_MIX)))
    lambda0_DP_vec = lambda0_DP * one_vec  # T_true
    lambda0 = jnp.concatenate((lambda0_MIX, lambda0_DP_vec), axis=1)  # J+T_true

    # NON cambio ora i nomi ma forse meglio non chiamare con 0 i parametri variazionali (eg lambda_0_mix o mu_0_mix)
    variational_parameters.nIW_MIX_VAR.labmdA= (lambda0 + sum_phi_k)[0,:J].T
    variational_parameters.nIW_DP_VAR.lambdA = (lambda0 + sum_phi_k)[0,J:].T


def update_NIW_nu(variational_parameters: VariationalParameters,hyperparameters_model: HyperparametersModel, sum_phi_k, T_true):
    J = hyperparameters_model.J
    nu0_DP = hyperparameters_model.nIW_DP_0.nu
    nu0_MIX = hyperparameters_model.nIW_MIX_0.nu

    one_vec = jnp.ones((1, T_true))

    # NU
    nu0_MIX = jnp.reshape(nu0_MIX, (1, len(nu0_MIX)))
    nu0_DP_vec = nu0_DP * one_vec  # T_true
    nu0 = jnp.concatenate((nu0_MIX, nu0_DP_vec), axis=1)  # J+T_true

    variational_parameters.nIW_MIX_VAR.nu = (nu0 + sum_phi_k)[0,:J]
    variational_parameters.nIW_DP_VAR.nu = (nu0 + sum_phi_k)[0,J:]

# modified until here

def update_NIW_PHI(variational_parameters: VariationalParameters,hyperparameters: HyperparametersModel, sum_phi_k, y_bar,y, phi_mk, T_true):
    J = hyperparameters.J
    M = hyperparameters.M
    #si accede così?
    mu0_DP = hyperparameters.nIW_DP_0.mu.T             # Txp
    mu0_MIX = hyperparameters.nIW_MIX_0.mu.T            # Jx1
    lambda0_DP = hyperparameters.nIW_DP_0.lambdA        # 1
    lambda0_MIX = hyperparameters.nIW_MIX_0.lambdA      # J
    PHI0_DP = hyperparameters.nIW_DP_0.phi              # 1xpxp
    PHI0_MIX = hyperparameters.nIW_MIX_0.phi            # Jxpxp

#    y_bar_matrix = jnp.repeat(y_bar[:, jnp.newaxis, :], M, axis=1)
    one_vec = jnp.ones((1, T_true))

    mu0_DP_vec = jnp.repeat(mu0_DP[:, :], T_true, axis=1)  # pxT_true
    mu0 = jnp.concatenate((mu0_MIX, mu0_DP_vec), axis=1)  #J+T_true x p

    lambda0_MIX = jnp.reshape(lambda0_MIX, (1, len(lambda0_MIX)))
    lambda0_DP_vec = lambda0_DP @ one_vec  # pxT_true
    lambda0 = jnp.concatenate((lambda0_MIX, lambda0_DP_vec), axis=1)  #J+T_true

    PHI0_DP_vec = jnp.repeat(PHI0_DP[jnp.newaxis, :, :], T_true, axis=0) #T_true x p x p
    PHI0 = jnp.concatenate((PHI0_MIX, PHI0_DP_vec), axis=0)  #J+T_true x p x p


    comp_2 = jnp.zeros(PHI0.shape)
    comp_3 = jnp.zeros(PHI0.shape)
    for k in range(J+T_true):
        curr_y_bar = y_bar[:, k] #Save the kth column of the matrix of the means
        y_bar_matrix = jnp.repeat(curr_y_bar[:, jnp.newaxis], M, axis=1) #Build a matrix where the columns are the same mean repeated M times
        diff_y = y.T-y_bar_matrix
        comp_2 = comp_2.at[k, :, :].set(diff_y @ (diff_y * phi_mk[:, k]).T) #Do the final operation, where phi_mk[:, k] is the M-dimensional column where we have the probs of being in the kth cluster

        diff = jnp.reshape(curr_y_bar - mu0[:, k], (1, len(curr_y_bar)))
        diff_matrix = diff.T @ diff
        coeff = lambda0[1, k] * sum_phi_k[k] / (lambda0[1, k]+sum_phi_k[k])
        comp_3 = comp_3.at[k, :, :].set(coeff*diff_matrix)

    variational_parameters.nIW_MIX_VAR.phi = (PHI0 + comp_2 + comp_3)[:J, :, :]
    variational_parameters.nIW_DP_VAR.phi = (PHI0 + comp_2 + comp_3)[J:, :, :]

def update_phi_mk(y, variational_parameters : VariationalParameters, T, J):
    p = y.shape[1]

    phi_mk = variational_parameters.phi_m_k
    eta_k = variational_parameters.eta_k
    mu_MIX = variational_parameters.nIW_MIX_VAR.mu
    lambda_MIX = variational_parameters.nIW_MIX_VAR.lambdA
    nu_MIX = variational_parameters.nIW_MIX_VAR.nu
    PHI_MIX = variational_parameters.nIW_MIX_VAR.phi

    a_k = variational_parameters.a_k_beta
    b_k = variational_parameters.b_k_beta
    mu_DP = variational_parameters.nIW_DP_VAR.mu
    lambda_DP = variational_parameters.nIW_DP_VAR.lambdA
    nu_DP = variational_parameters.nIW_DP_VAR.nu
    PHI_DP = variational_parameters.nIW_DP_VAR.phi

    eta_signed = jnp.sum(eta_k)
    l = 1 - jnp.array(range(1, p+1))

    for k in range(J):
        e_dir = digamma(eta_k[0,k]) - digamma(eta_signed)

        e_norm = -1/2 * (-jnp.sum(digamma((nu_MIX[k] - l)/2)) + jnp.log(jdet(PHI_MIX[k, :, :]))
                         + p/lambda_MIX[k] + nu_MIX[k] * ((y - mu_MIX[k,0]) @ jinv(PHI_MIX[k, :, :]) @
                         (y - mu_MIX[k,0]).T))[0,:]

        phi_mk = phi_mk.at[:, k].set(jnp.exp(e_dir + e_norm))

    e_dir = digamma(eta_k[0,0]) - digamma(eta_signed)

    dig_b = digamma(b_k)
    dig_a = digamma(a_k + b_k)
    dig_diff = dig_b - dig_a
    diff = jnp.cumsum(dig_diff)
    diff = jnp.reshape(diff, (1,len(diff)))
    z = jnp.zeros((1,1))
    dig_cumsum = jnp.concatenate((z, diff), axis=1)

    for k in range(T):
        e_beta = digamma(a_k[k]) + digamma(a_k[k] + b_k[k])

        e_res = dig_cumsum[0,k]

        e_norm = -1 / 2 * (-jnp.sum(digamma((nu_DP[k] - l) / 2))
                           + jnp.log(jdet(PHI_DP[k, :, :]))
                           + p / lambda_DP[k] + nu_DP[k]
                           * ((y - mu_DP[0,k]) @ jinv(PHI_DP[k, :, :]) @ (
                                       y - mu_DP[0,k].T).T))[0,:]

        phi_mk = phi_mk.at[:, k+J].set(jnp.exp(e_dir + e_beta + e_res + e_norm))

    norm_phi = jnp.reshape(jnp.sum(phi_mk, axis=1), (phi_mk.shape[0], 1))
    variational_parameters.phi_m_k = phi_mk / norm_phi



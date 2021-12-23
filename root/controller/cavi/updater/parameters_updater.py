from jax.numpy.linalg import det as jdet
from jax.numpy.linalg import inv as jinv
from jax.scipy.special import digamma
from jax.scipy.special import logsumexp
from jax import numpy as jnp
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters
import numpy as np

# Use jit and vectorization!

def update_parameters(data, hyperparameters: HyperparametersModel, variational_parameters: VariationalParameters, starting_parameters : VariationalParameters):
    #update parameters

    # print('mu = ', variational_parameters.nIW.mu)
    # print('lambda = ', variational_parameters.nIW.lambdA)
    # print('nu = ', variational_parameters.nIW.nu)
    # print('PHI = ', variational_parameters.nIW.phi)
    # print('a_beta = ', variational_parameters.a_k_beta)
    # print('b_beta = ', variational_parameters.b_k_beta)
    # print('phi_m = ', variational_parameters.phi_m_k)
    # print('eta = ', variational_parameters.eta_k)

    # Define function for each family update:
    # Multinomial_phi
    variational_parameters.phi_m_k = update_phi_mk(data, variational_parameters, hyperparameters.T, hyperparameters.J)

    sum_phi_k = jnp.sum(variational_parameters.phi_m_k, axis=0)
    # print(sum_phi_k)
    # Dirichlet_eta
    variational_parameters.eta_k = update_dirichlet(hyperparameters, sum_phi_k)
    # Beta_V
    variational_parameters.a_k_beta, variational_parameters.b_k_beta = update_beta(hyperparameters, sum_phi_k)
    # NIW_mu_nu_lamda_Phi
    update_NIW(data, starting_parameters, variational_parameters, hyperparameters, sum_phi_k)

    #update jax array
    #return variational_parameters


############# UTILS ##############
def eval_y_bar(sum_phi_k, sum_y_phi):
    y_bar = sum_y_phi * 1/sum_phi_k            #(px(J+T_true))*((J+T_true)x(J+T_true)) = px(J+T_true)
    return y_bar

def create_tensor(diff, hyperparameters):
    #Build a 3D pxMxM tensor where we have a p-dimensional array from diff on each element of the MxM matrix
    transf = jnp.eye(hyperparameters.M)
    # transf = jnp.eye(2)
    transf = jnp.repeat(transf[:, :, np.newaxis], hyperparameters.p, axis=2)

    return diff * transf


############# UPDATE DIRICHLET ##############
def update_dirichlet(hyperparameters : HyperparametersModel, sum_phi_k):
    J = hyperparameters.J
    temporary_phi_k = jnp.zeros(J+1)
    temporary_phi_k = temporary_phi_k.at[1:(J+1)].set(sum_phi_k[:J])
    temporary_phi_k = temporary_phi_k.at[0].set(jnp.sum(sum_phi_k[J:]))

    return hyperparameters.a_dir_k + temporary_phi_k


############# UPDATE BETA ##############
def update_beta(hyperparameters : HyperparametersModel, sum_phi_k):
    J = hyperparameters.J
    T = hyperparameters.T

    remaining_probs = jnp.cumsum(jnp.flip(sum_phi_k[(J+1):]))
    remaining_probs = jnp.flip(remaining_probs)

    # print("remaining probabilities = ", remaining_probs)

    return sum_phi_k[J:J+T-1] + 1, hyperparameters.gamma + remaining_probs


###############################################################
###############          NIW                ###################
###############################################################

############# UPDATE NIW ##############à
def update_NIW(y, starting_parameters : VariationalParameters,
               variational_parameters : VariationalParameters,
               hyperparameters  : HyperparametersModel,
               sum_phi_k):
    phi_mk = variational_parameters.phi_m_k

    # y = jnp.array([[1, 0], [0, 1]])
    # phi_mk = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    # sum_phi_k = jnp.sum(phi_mk, axis=0)

    # starting_parameters.nIW.mu = jnp.reshape(jnp.array([1, 1]), (1,2))
    # starting_parameters.nIW.lambdA = jnp.array([1])
    # starting_parameters.nIW.nu = jnp.array([1])
    # starting_parameters.nIW.phi = jnp.reshape(jnp.array([[1,1],[1,2]]), (1,2,2))

    # supponendo y Mxp
    sum_y_phi = y.T @ phi_mk                         # pxM*(M*(J+T_true)) = px(J+T_true)
    y_bar = eval_y_bar(sum_phi_k, sum_y_phi)                # px(J+T_true)

    variational_parameters.nIW.lambdA = update_NIW_lambda(starting_parameters, sum_phi_k)
    variational_parameters.nIW.nu = update_NIW_nu(starting_parameters, sum_phi_k)
    variational_parameters.nIW.mu = update_NIW_mu(starting_parameters, variational_parameters, sum_y_phi)
    variational_parameters.nIW.phi = update_NIW_PHI(y, starting_parameters, variational_parameters, hyperparameters,
                                                    sum_phi_k, y_bar, phi_mk)

    # print('mu = ', variational_parameters.nIW.mu)
    # print('lambda = ', variational_parameters.nIW.lambdA)
    # print('nu = ', variational_parameters.nIW.nu)
    # print('PHI = ', variational_parameters.nIW.phi)


def update_NIW_lambda(starting_parameters : VariationalParameters,
                      sum_phi_k):
    lambda0 = starting_parameters.nIW.lambdA

    return lambda0 + sum_phi_k


###TEST NIW
# mu_0 = jnp.array([1,1])[None,:]
#lam_0 = jnp.array([1])
#nu_0 = jnp.array([1])
#phi_0 = jnp.array([[1,1],[1,2])[None,:]
#
#data = jnp.array([[1,0],[0,1]])
#phi_m_k = jnp.array([[0.5, 0.5],[0.5,0.5])
#
#
#EXPECTED
#mu_k = [3/4, 3/4]
#lambda_k = 2
#nu_k = 2
#phi_k =    [13/8    5/8]
#           [5/8    21/8]


def update_NIW_mu(starting_parameters : VariationalParameters,
                  variational_parameters : VariationalParameters,
                  sum_y_phi):
    # Estrazione parametri
    lambda0 = starting_parameters.nIW.lambdA
    mu0 = starting_parameters.nIW.mu.T      # px(J+T)

    # CALCOLI AGGIORNAMENTO
    num = lambda0 * mu0 + sum_y_phi      # px(J+T_true)
    den = variational_parameters.nIW.lambdA
    mu_k = num/den   # px(J+T_true)

    return mu_k.T


def update_NIW_nu(starting_parameters : VariationalParameters,
                  sum_phi_k):
    nu0 = starting_parameters.nIW.nu

    return nu0 + sum_phi_k


def update_NIW_PHI(y, starting_parameters : VariationalParameters,
                   variational_parameters : VariationalParameters,
                   hyperparameters: HyperparametersModel,
                   sum_phi_k, y_bar, phi_mk):
    J = hyperparameters.J
    T = hyperparameters.T
    # J = 1
    # T = 1
    mu0 = starting_parameters.nIW.mu.T           # px(J+T)
    lambda0 = starting_parameters.nIW.lambdA     # J+T
    PHI0 = starting_parameters.nIW.phi           # (J+T)xpxp
    lambdA = variational_parameters.nIW.lambdA

    comp_2 = jnp.zeros(PHI0.shape)
    comp_3 = jnp.zeros(PHI0.shape)
    for k in range(J+T):
        curr_y_bar = y_bar[:, k:(k+1)]  # Save the kth column of the matrix of the means    px1
        diff_y = y.T - curr_y_bar

        # alternative code todo check
        tensor_y = create_tensor(diff_y.T, hyperparameters)
        diff_y *= phi_mk[:, k]
        prod = jnp.dot(diff_y, jnp.transpose(tensor_y, (2, 0, 1))) # (pxM)x(pxMxM) = pxpxM

        comp_2 = comp_2.at[k, :, :].set(jnp.sum(prod, axis = 2))

        diff = jnp.reshape(curr_y_bar.T - mu0[:, k], (1, len(curr_y_bar)))     # 1xp

        diff_matrix = diff.T @ diff              # pxp

        coeff = lambda0[k] * sum_phi_k[k] / lambdA[k]          #1

        comp_3 = comp_3.at[k, :, :].set(coeff*diff_matrix)                      # (J+T)xpxp

    # print('component 2 = ', comp_2)
    # print('component 3 = ', comp_3)

    return PHI0 + comp_2 + comp_3

######## UPDATE CATEGORIAL ########


def update_phi_mk(y, variational_parameters : VariationalParameters, T : int, J : int):
    p = y.shape[1]

    phi_mk = variational_parameters.phi_m_k
    eta_k = variational_parameters.eta_k
    mu = variational_parameters.nIW.mu
    lambdA = variational_parameters.nIW.lambdA
    nu = variational_parameters.nIW.nu
    PHI = variational_parameters.nIW.phi
    a_k = variational_parameters.a_k_beta
    b_k = variational_parameters.b_k_beta

    eta_signed = jnp.sum(eta_k)
    l = 1 - jnp.arange(1, p+1)

    e_dir = digamma(eta_k) - digamma(eta_signed)

    for k in range(J):
        e_norm = -1/2 * (-jnp.sum(digamma((nu[k] - l)/2)) + jnp.log(jdet(PHI[k, :, :]))
                         + p/lambdA[k] + nu[k] * jnp.diag((y - mu[k, :]) @ jinv(PHI[k, :, :]) @ (y - mu[k,:]).T))

        phi_mk = phi_mk.at[:, k].set(jnp.exp(e_dir[k+1] + e_norm))

    dig_b = digamma(b_k)
    dig_a = digamma(a_k + b_k)
    dig_diff = dig_b - dig_a
    diff = jnp.cumsum(dig_diff)

    diff = jnp.reshape(diff, (1,len(diff)))

    z = jnp.zeros((1,1))
    dig_cumsum = jnp.concatenate((z, diff), axis=1)

    for k in range(0, T-1):
        e_beta = digamma(a_k[k]) - digamma(a_k[k] + b_k[k])

        e_res = dig_cumsum[0,k]

        e_norm = -1 / 2 * (-jnp.sum(digamma((nu[k+J] - l) / 2))
                           + jnp.log(jdet(PHI[k+J, :, :]))
                           + p / lambdA[k+J]
                           + nu[k+J] * jnp.diag(((y - mu[k+J, :]) @ jinv(PHI[k+J, :, :]) @ (
                                       y - mu[k+J, :]).T)))

        phi_mk = phi_mk.at[:, k+J].set(jnp.exp(e_dir[0] + e_beta + e_res + e_norm))

    e_res_T = dig_cumsum[0, T-1]

    e_norm_T = -1 / 2 * (-jnp.sum(digamma((nu[J+T-1] - l) / 2))
                       + jnp.log(jdet(PHI[(J+T-1), :, :]))
                       + p / lambdA[J+T-1] + nu[J+T-1]
                       * jnp.diag(((y - mu[J+T-1,:]) @ jinv(PHI[J+T-1, :, :]) @ (
                    y - mu[J+T-1,:]).T)))

    phi_mk = phi_mk.at[:, T + J - 1].set(jnp.exp(e_dir[0] + e_res_T + e_norm_T))
    norm_phi = jnp.reshape(jnp.sum(phi_mk, axis=1), (phi_mk.shape[0], 1))

    return phi_mk / norm_phi





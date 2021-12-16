# from jax._src.numpy.linalg import jdet, jinv
# from jax._src.scipy.special import jdigamma as digamma
from jax.numpy.linalg import det as jdet, inv as jinv
from jax.scipy.special import digamma as digamma
from jax.scipy.special import digamma as jdigamma
from jax.scipy.special import gammaln as jgammaln

from jax import numpy as jnp
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters
import numpy as np

# Use jit and vectorization!

def init_update_parameters(hyperparameters: HyperparametersModel, variational_parameters: VariationalParameters):
    variational_parameters.sum_phi_k = jnp.sum(variational_parameters.phi_m_k, axis=0)
    # print(sum_phi_k.shape)

    # mask = create_mask(sum_phi_k, J)
    # sum_phi_k = sum_phi_k[mask]
    variational_parameters.T_true = len(variational_parameters.sum_phi_k) - hyperparameters.J

def update_parameters(data, hyperparameters: HyperparametersModel, variational_parameters: VariationalParameters):
    # Define function for each family update:
    # Dirichlet_eta
    update_dirichlet(variational_parameters, hyperparameters)
    # Beta_V
    update_beta(variational_parameters, hyperparameters)
    # NIW_mu_nu_lamda_Phi
    init_update_NIW(data, variational_parameters)
    update_NIW(data, variational_parameters, hyperparameters)
    # Multinomial_phi
    update_phi_mk(data, variational_parameters, hyperparameters)

    # #update jax array
    # return variational_parameters


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
def update_dirichlet(variational_parameters : VariationalParameters, hyperparameters):
    J = hyperparameters.J
    temporary_phi_k = jnp.zeros(J+1)
    temporary_phi_k = temporary_phi_k.at[0:-1].set(variational_parameters.sum_phi_k[:J])
    temporary_phi_k = temporary_phi_k.at[-1].set(jnp.sum(variational_parameters.sum_phi_k[J:]))

    variational_parameters.eta_k = hyperparameters.a_dir_k + temporary_phi_k


############# UPDATE BETA ##############
def update_beta(variational_parameters : VariationalParameters, hyperparameters : HyperparametersModel):
    J = hyperparameters.J
    T = hyperparameters.T

    remaining_probs = jnp.cumsum(jnp.flip(variational_parameters.sum_phi_k[J:]))
    remaining_probs = jnp.flip(remaining_probs)

    # print(remaining_probs)

    variational_parameters.a_k_beta = variational_parameters.a_k_beta.at[0:T-1].set(jnp.add(variational_parameters.sum_phi_k[J:J+T-1], 1))
    variational_parameters.b_k_beta = variational_parameters.b_k_beta.at[0:T-1].set(jnp.add(hyperparameters.gamma, remaining_probs[0:T-1]))



###############################################################
###############          NIW                ###################
###############################################################

############# UPDATE NIW ##############à
def init_update_NIW(y, variational_parameters : VariationalParameters):
    # supponendo y Mxp
    variational_parameters.sum_y_phi = y.T @ variational_parameters.phi_m_k                         # pxM*(M*(J+T_true)) = px(J+T_true)
    variational_parameters.y_bar = eval_y_bar(variational_parameters.sum_phi_k, variational_parameters.sum_y_phi)                # px(J+T_true)

def update_NIW(y, variational_parameters : VariationalParameters, hyperparameters: HyperparametersModel):
    update_NIW_mu(variational_parameters, hyperparameters)
    update_NIW_lambda(variational_parameters, hyperparameters)
    update_NIW_nu(variational_parameters, hyperparameters)
    update_NIW_PHI(y, variational_parameters, hyperparameters)

def update_NIW_mu(variational_parameters: VariationalParameters , hyperparameters_model: HyperparametersModel):
    # Estrazione parametri
    J = hyperparameters_model.J
    lambda0_DP = hyperparameters_model.nIW_DP_0.lambdA
    lambda0_MIX = hyperparameters_model.nIW_MIX_0.lambdA
    mu0_DP = hyperparameters_model.nIW_DP_0.mu.T           # pxT
    mu0_MIX = hyperparameters_model.nIW_MIX_0.mu.T        # pxJ

    one_vec = jnp.ones((1, variational_parameters.T_true))

    # LAMBDA
    lambda0_MIX = jnp.reshape(lambda0_MIX, (1, len(lambda0_MIX)))
    lambda0_DP_vec = lambda0_DP * one_vec  # T_true
    lambda0 = jnp.concatenate((lambda0_MIX, lambda0_DP_vec), axis=1)          # J+T_true

    # MU
    mu0_DP_vec = jnp.repeat(mu0_DP, variational_parameters.T_true, axis=1)
    # pxT_true
    mu0 = jnp.concatenate((mu0_MIX, mu0_DP_vec), axis=1)        # px(J+T_tr

    # CALCOLI AGGIORNAMENTO
    num = lambda0 * mu0 + variational_parameters.sum_y_phi      # px(J+T_true)
    den = lambda0 + variational_parameters.sum_phi_k
    mu_k = num/den                                                          # px(J+T_true)

    variational_parameters.nIW_MIX_VAR.mu = mu_k[:, :J].T
    variational_parameters.nIW_DP_VAR.mu = mu_k[:, J:].T


def update_NIW_lambda(variational_parameters: VariationalParameters , hyperparameters_model: HyperparametersModel):
    J = hyperparameters_model.J
    lambda0_DP = hyperparameters_model.nIW_DP_0.lambdA
    lambda0_MIX = hyperparameters_model.nIW_MIX_0.lambdA

    one_vec = jnp.ones((1, variational_parameters.T_true))

    # LAMBDA
    lambda0_MIX = jnp.reshape(lambda0_MIX, (1, len(lambda0_MIX)))
    lambda0_DP_vec = lambda0_DP * one_vec  # T_true
    lambda0 = jnp.concatenate((lambda0_MIX, lambda0_DP_vec), axis=1)  # J+T_true

    # NON cambio ora i nomi ma forse meglio non chiamare con 0 i parametri variazionali (eg lambda_0_mix o mu_0_mix)

    variational_parameters.nIW_MIX_VAR.labmdA= (lambda0 + variational_parameters.sum_phi_k)[0,:J].T
    variational_parameters.nIW_DP_VAR.lambdA = (lambda0 + variational_parameters.sum_phi_k)[0,J:].T


def update_NIW_nu(variational_parameters: VariationalParameters,hyperparameters_model: HyperparametersModel):
    J = hyperparameters_model.J
    nu0_DP = hyperparameters_model.nIW_DP_0.nu
    nu0_MIX = hyperparameters_model.nIW_MIX_0.nu

    one_vec = jnp.ones((1, variational_parameters.T_true))

    # NU
    nu0_MIX = jnp.reshape(nu0_MIX, (1, len(nu0_MIX)))
    nu0_DP_vec = nu0_DP * one_vec  # T_true
    nu0 = jnp.concatenate((nu0_MIX, nu0_DP_vec), axis=1)  # J+T_true

    variational_parameters.nIW_MIX_VAR.nu = (nu0 + variational_parameters.sum_phi_k)[0,:J]
    variational_parameters.nIW_DP_VAR.nu = (nu0 + variational_parameters.sum_phi_k)[0,J:]


# modified until here

def update_NIW_PHI(y, variational_parameters: VariationalParameters,hyperparameters: HyperparametersModel):
    J = hyperparameters.J
    M = hyperparameters.M
    #si accede così?
    mu0_DP = hyperparameters.nIW_DP_0.mu.T              # px1
    mu0_MIX = hyperparameters.nIW_MIX_0.mu.T            # pxJ
    lambda0_DP = hyperparameters.nIW_DP_0.lambdA        # 1
    lambda0_MIX = hyperparameters.nIW_MIX_0.lambdA      # J
    PHI0_DP = hyperparameters.nIW_DP_0.phi              # 1xpxp
    PHI0_MIX = hyperparameters.nIW_MIX_0.phi            # Jxpxp

    # print(mu0_DP.shape)
    # print(mu0_MIX.shape)
    # print(lambda0_DP.shape)
    # print(lambda0_MIX.shape)
    # print(PHI0_DP.shape)
    # print(PHI0_MIX.shape)

#    y_bar_matrix = jnp.repeat(y_bar[:, jnp.newaxis, :], M, axis=1)
    one_vec = jnp.ones((1, variational_parameters.T_true))

    mu0_DP_vec = jnp.repeat(mu0_DP[:, :], variational_parameters.T_true, axis=1)  # pxT_true
    mu0 = jnp.concatenate((mu0_MIX, mu0_DP_vec), axis=1)  # pxJ+T_true

    # print(mu0_DP_vec.shape)
    # print(mu0.shape)

    #lambda0_MIX = jnp.reshape(lambda0_MIX, (1, len(lambda0_MIX)))
    lambda0_DP_vec = lambda0_DP @ one_vec  # 1xT_true
    lambda0 = jnp.concatenate((lambda0_MIX, lambda0_DP_vec))  #1xJ+T_true

    # print(lambda0_DP_vec.shape)
    # print(lambda0.shape)


    PHI0_DP_vec = jnp.repeat(PHI0_DP[jnp.newaxis, :, :], variational_parameters.T_true, axis=0) #T_true x p x p
    PHI0 = jnp.concatenate((PHI0_MIX, PHI0_DP_vec), axis=0)  #J+T_true x p x p

    # print(PHI0_DP_vec.shape)
    # print(PHI0.shape)

    comp_2 = jnp.zeros(PHI0.shape)
    comp_3 = jnp.zeros(PHI0.shape)
    for k in range(J+variational_parameters.T_true):
        curr_y_bar = variational_parameters.y_bar[:, k] #Save the kth column of the matrix of the means    px1
        y_bar_matrix = jnp.repeat(curr_y_bar[:, jnp.newaxis], M, axis=1) #Build a matrix where the columns are the same mean repeated M times
        diff_y = y.T-y_bar_matrix
        # comp_2 = comp_2.at[k, :, :].set(diff_y @ (diff_y * phi_mk[:, k]).T) #Do the final operation, where phi_mk[:, k] is the M-dimensional column where we have the probs of being in the kth cluster


        # for m in range(M):
        #     vector = diff_y[:, m]
        #     comp_2 = comp_2.at[k, :, :].set(comp_2[k,:,:] + vector @ vector.T * phi_mk[m, k])

        # alternative code todo check
        tensor_y = create_tensor(diff_y.T,hyperparameters)
        #print(diff_y.shape)
        #print(tensor_y.shape)
        diff_y *= variational_parameters.phi_m_k[:,k]
        prod = jnp.dot(diff_y, jnp.transpose(tensor_y, (2,0,1))) # (pxM)x(pxMxM) = pxpxM
        #print(prod.shape)
        comp_2 = comp_2.at[k, :, :].set(jnp.sum(prod, axis = 2))

        #res = jnp.sum(prod, axis=2)

        # print('comp_2', comp_2)
        # print('comp_2, shape', comp_2.shape)

        # diff_y = diff_y[:, jnp.newaxis, :]
        # diff_y_T = jnp.transpose(diff_y, axes =  (1,2,0))
        # diff_y = diff_y * phi_mk[:, k]

        # print(diff_y.shape)
        # print(diff_y_T.shape)
        # print(jnp.dot(diff_y,diff_y_T ).shape)
        # # (2, 1, 500)
        # (1, 500, 2)
        # (2, 1, 1, 2)

        # print((diff_y @ diff_y_T).shape)
        # (2, 1, 500)
        # (1, 500, 2)
        # (2, 1, 2)

        # comp_2 = comp_2.at[k, :, :].set(jnp.sum( diff_y @ diff_y_T, axis = 2 ))

        # print((curr_y_bar - mu0[:, k]).shape)
        # (curr_y_bar - mu0[:, k]).shape = (p,)
        diff = jnp.reshape(curr_y_bar - mu0[:, k], (1, len(curr_y_bar)))     # 1xp

        # print(diff.shape)

        diff_matrix = diff.T @ diff              # pxp

        # print(diff_matrix.shape)

        coeff = lambda0[k] * variational_parameters.sum_phi_k[k] / (lambda0[k]+variational_parameters.sum_phi_k[k])           #1

        # print(coeff.shape)

        comp_3 = comp_3.at[k, :, :].set(coeff*diff_matrix)                      # (J+T)xpxp

        # print(comp_3.shape)


    variational_parameters.nIW_MIX_VAR.phi = (PHI0 + comp_2 + comp_3)[:J, :, :]
    variational_parameters.nIW_DP_VAR.phi = (PHI0 + comp_2 + comp_3)[J:, :, :]

    # print(jdet(variational_parameters.nIW_MIX_VAR.phi))
    # print(jdet(variational_parameters.nIW_DP_VAR.phi))


def update_phi_mk(y, variational_parameters : VariationalParameters, hyperparameter: HyperparametersModel):
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

    #print(variational_parameters.phi_m_k)

    eta_signed = jnp.sum(eta_k)
    l = 1 - jnp.array(range(1, p+1))

    # out =  (y - mu_MIX[0,:]) @ jinv(PHI_MIX[0, :, :]) @ (y - mu_MIX[0,:]).T
    #print( 'MATRICIONAAA',out )
    #print('MATRICIONAAA', out.shape)
    # out = ((y - mu_MIX[0,:]) @ jinv(PHI_MIX[0, :, :]) @ (y - mu_MIX[0,:]).T)
    #print( 'MATRICIONAAA',y)

    # out = ((y - mu_MIX[0,:]).T @ jinv(PHI_MIX[0, :, :]) @ (y - mu_MIX[0,:]))
    # print( 'scalare?',out)
    J = hyperparameter.J

    for k in range(J):
        e_dir = digamma(eta_k[k]) - digamma(eta_signed)

        e_norm = -1/2 * (-jnp.sum(digamma((nu_MIX[k] - l)/2)) + jnp.log(jdet(PHI_MIX[k, :, :]))
                         + p/lambda_MIX[k] + nu_MIX[k] * jnp.diag( (y - mu_MIX[k,:]) @ jinv(PHI_MIX[k, :, :]) @ (y - mu_MIX[k,:]).T) )
    # Mxp-1xp
                                                                        # Mxp
# problema matrice MxM

        # mu__MIX -> matrice (Jxp)
        #           -> riga per riga ci sono le medie delle componenti della mistura
        #               (mu_0_MIX[i,:] -> media della (i+1)-esima NIW della mistura)

        phi_mk = phi_mk.at[:, k].set(jnp.exp(e_dir + e_norm))

    e_dir = digamma(eta_k[0]) - digamma(eta_signed)
    # print('digammavec', digamma((nu_MIX[0] - l)/2))

    dig_b = digamma(b_k)
    dig_a = digamma(a_k + b_k)
    dig_diff = dig_b - dig_a
    diff = jnp.cumsum(dig_diff)
#    print(diff.shape)

    diff = jnp.reshape(diff, (1,len(diff)))
    # print(diff.shape)

    z = jnp.zeros((1,1))
    dig_cumsum = jnp.concatenate((z, diff), axis=1)

    T = hyperparameter.T
    J = hyperparameter.J

    for k in range(0,T-1):
        e_beta = digamma(a_k[k]) + digamma(a_k[k] + b_k[k])
        # todo
        # Questo è coerente con la formula?
        e_res = dig_cumsum[0,k]

        e_norm = -1 / 2 * (-jnp.sum(digamma((nu_DP[k] - l) / 2))
                           + jnp.log(jdet(PHI_DP[k, :, :]))
                           + p / lambda_DP[k] + nu_DP[k]
                           * jnp.diag( ((y - mu_DP[k,:]) @ jinv(PHI_DP[k, :, :]) @ (
                                       y - mu_DP[k,:]).T)))

        phi_mk = phi_mk.at[:, k+J].set(jnp.exp(e_dir + e_beta + e_res + e_norm))

    e_res_T = dig_cumsum[0, T-1]

    e_norm_T = -1 / 2 * (-jnp.sum(digamma((nu_DP[T-1] - l) / 2))
                       + jnp.log(jdet(PHI_DP[(T-1), :, :]))
                       + p / lambda_DP[T-1] + nu_DP[T-1]
                       * jnp.diag( ((y - mu_DP[T-1,:]) @ jinv(PHI_DP[T-1, :, :]) @ (
                    y - mu_DP[T-1,:]).T) ))

    phi_mk = phi_mk.at[:, T + J-1].set(jnp.exp(e_dir + e_res_T + e_norm_T))
    norm_phi = jnp.reshape(jnp.sum(phi_mk, axis=1), (phi_mk.shape[0], 1))

    # print('norm_phi', norm_phi.shape)
    #
    # print('2', variational_parameters.phi_m_k.shape)

    variational_parameters.phi_m_k = phi_mk / norm_phi

    # print('2',variational_parameters.phi_m_k.shape)

def create_tensor_wrong(phi_mk):
    expanded = np.zeros(phi_mk.shape + phi_mk.shape[-1:], dtype=phi_mk.dtype)
    diagonals = np.diagonal(expanded, axis1=-2, axis2=-1)
    diagonals.setflags(write=True)
    diagonals[:] = phi_mk
    return jnp.array(expanded)

def create_tensor(diff, hyperparameters):
     #
    temp = jnp.repeat(diff[:, :, jnp.newaxis], hyperparameters.M, axis=2)

    transf = np.eye(hyperparameters.M)
    transf = np.repeat(transf[:, :, np.newaxis], hyperparameters.p, axis=2)

    return jnp.transpose(temp, (0,2,1)) * transf


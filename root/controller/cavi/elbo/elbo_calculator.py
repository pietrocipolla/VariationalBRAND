from jax.scipy.special import digamma as jdgamma
from jax.scipy.special import gammaln as jgammaln
from jax import numpy as jnp
from root.controller.cavi.utils import useful_functions
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters

def elbo_calculator(data, hyper: HyperparametersModel, var_param: VariationalParameters, p, psi_dp=None):
    M = hyper.M
    J = hyper.J
    T = hyper.T

    gamma = hyper.gamma
    a_dir_k = hyper.a_dir_k
    nIW_DP_0 = hyper.nIW_DP_0
    nIW_MIX_0 = hyper.nIW_MIX_0

    phi_m_k = var_param.phi_m_k
    eta_k = var_param.eta_k
    a_k_beta = var_param.a_k_beta
    b_k_beta = var_param.b_k_beta
    nIW = var_param.nIW

    eta_bar = jnp.sum(eta_k)

    mu_mix = nIW.mu[:J, :]
    nu_mix = nIW.nu[:J]
    lam_mix = nIW.lambdA[:J]
    psi_mix = nIW.phi[:J, :, :]

    # print('psi_mix')
    # print(psi_mix)

    mu_dp = nIW.mu[J:, :]
    nu_dp = nIW.nu[J:]
    lam_dp = nIW.lambdA[J:]
    psi_dp = nIW.phi[J:, :]

    diga_a = jdgamma(a_k_beta)
    diga_b = jdgamma(b_k_beta)
    diga_ab = jdgamma(a_k_beta + b_k_beta)
    diga_e_b = jdgamma(eta_bar)
    diga_eta = jdgamma(eta_k)

    l = 1 - jnp.array(range(1, p + 1))

    # print('psi_dp')
    # print(psi_dp)

    #val atteso log p

    # #f1 e f2
    f1=0
    f2=0

    for k in range(0, J):
        f1 += useful_functions.E_log_norm(phi_m_k[:,k], data, mu_mix[k], nu_mix[k], lam_mix[k], psi_mix[k], p, l, M)
    for k in range(0, T):
        f2 += useful_functions.E_log_norm(phi_m_k[:,k+J], data, mu_dp[k], nu_dp[k], lam_dp[k], psi_dp[k], p, l, M)

    print('f1 f2 done',f1,f2)
    #
    #
    # #f3 e f4
    f3=0
    f4=0
    for k in range(0,J):
        f3 += useful_functions.E_log_dens_norm_inv_wish_p(mu_mix[k,:],nu_mix[k],lam_mix[k],psi_mix[k,:,:],nIW_MIX_0.mu[k,:], nIW_MIX_0.nu[k], nIW_MIX_0.lambdA[k], nIW_MIX_0.phi[k,:,:] ,p,l)
    for k in range(0,T):
        f4 += useful_functions.E_log_dens_norm_inv_wish_p(mu_dp[k, :], nu_dp[k], lam_dp[k], psi_dp[k, :, :],
                                                          nIW_DP_0.mu, nIW_DP_0.nu, nIW_DP_0.lambdA, nIW_DP_0.phi, p, l)
    print('f3 f4 done',f3,f4)

    f5 = jnp.sum(jnp.multiply(jnp.sum(phi_m_k,axis = 0)[0:J], (diga_eta - diga_e_b)[1:]))

    f6 = 0
#
    #print('f6 done', f6)

    #
    parsum = jnp.cumsum(diga_b - diga_ab)

    f6 = jnp.sum(phi_m_k[:, J]) * (diga_eta[0] - diga_e_b + diga_a[0] - diga_ab[0])

    f6 += jnp.sum(jnp.multiply(jnp.sum(phi_m_k[:, (J+1):(J+T-1)],axis = 0), (diga_eta[0] - diga_e_b + diga_a[1:(T-1)] - diga_ab[1:(T-1)] + parsum[0:(T-2)])))
    f6 += jnp.sum(phi_m_k[:, J + T - 1]) * (diga_eta[0] - diga_e_b + parsum[T - 2])
    print('f6 done', f6)
    #

    f7 = jnp.sum(jnp.multiply((a_dir_k-1),(diga_eta-diga_e_b)))
    #
    print('f7 done', f7)

    f8 = jnp.sum(jnp.multiply((gamma-1),(diga_b-diga_ab)))
    print('f8 done',f8)

    E_log_p = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8

    #val atteso log q

    h1 = jnp.sum(jnp.log(phi_m_k**phi_m_k))
    print('h1 done', h1)

    #h2
    h2=0
    h2 = jnp.sum(jnp.multiply((eta_k-1),diga_eta-diga_e_b))
    h2 += jgammaln(eta_bar)-jnp.sum(jgammaln(eta_k))
    print('h2 done', h2)

    #h3
    h3 = 0
    beta_AB = jgammaln(a_k_beta) + jgammaln(b_k_beta) - jgammaln(a_k_beta + b_k_beta)
    h3 = jnp.sum(jnp.multiply(a_k_beta-1,diga_a - diga_ab)) + jnp.sum(jnp.multiply(b_k_beta-1,diga_b - diga_ab)) - jnp.sum(beta_AB)
    print('h3 done', h3)

    #h4 e h5
    h4=0
    for k in range(0,J):
        h4 += useful_functions.E_log_dens_norm_inv_wish_q(mu_mix[k,:],nu_mix[k],lam_mix[k],psi_mix[k,:,:],p,l)
    h5 = 0
    for k in range(0, T):
        h5 += useful_functions.E_log_dens_norm_inv_wish_q(mu_dp[k,:],nu_dp[k],lam_dp[k],psi_dp[k,:,:],p,l)

    E_log_q= h1+h2+h3+h4+h5
    print('h4 h5 done', h4,h5)

    return E_log_p-E_log_q

from jax.numpy import log as jlog
# from jax._src.scipy.special import jdigamma as jdgamma
# from jax._src.scipy.special import gammaln as jgammaln
from jax.scipy.special import digamma as jdgamma
from jax.scipy.special import gammaln as jgammaln
from jax import numpy as jnp
from root.controller.cavi.utils import useful_functions
from root.model.hyperparameters_model import HyperparametersModel
from root.model.variational_parameters import VariationalParameters


def elbo_calculator(data, hyper: HyperparametersModel, var_param: VariationalParameters, p, psi_dp=None):
    M=hyper.M
    J=hyper.J
    T=hyper.T

    gamma=hyper.gamma
    a_dir_k=hyper.a_dir_k
    nIW_DP_0=hyper.nIW_DP_0
    nIW_MIX_0=hyper.nIW_MIX_0

    phi_m_k=var_param.phi_m_k
    eta_k=var_param.eta_k
    a_k_beta=var_param.a_k_beta
    b_k_beta=var_param.b_k_beta
    nIW_MIX_VAR=var_param.nIW_MIX_VAR
    nIW_DP_VAR=var_param.nIW_DP_VAR

    eta_bar=jnp.sum(eta_k)

    mu_mix=nIW_MIX_VAR.mu
    nu_mix=nIW_MIX_VAR.nu
    lam_mix=nIW_MIX_VAR.lambdA
    psi_mix=nIW_MIX_VAR.phi

    # print('psi_mix')
    # print(psi_mix)

    mu_dp=nIW_DP_VAR.mu
    nu_dp=nIW_DP_VAR.nu
    lam_dp=nIW_DP_VAR.lambdA
    psi_dp=nIW_DP_VAR.phi

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
    #for m in range(0,M):
    #    for k in range(0,J):
    #        f1 += phi_m_k[m,k]*useful_functions.E_log_norm(data[m,:],mu_mix[k],nu_mix[k],lam_mix[k],psi_mix[k,:,:],p, l)
    #    for k in range(0,T):
    #        f2 += phi_m_k[m,k+J]*useful_functions.E_log_norm(data[m,:],mu_dp[k],nu_dp[k],lam_dp[k],psi_dp[k,:,:],p, l)
    for k in range(0, J):
        f1 += useful_functions.E_log_norm(phi_m_k[:,k], data, mu_mix[k], nu_mix[k], lam_mix[k], psi_mix[k], p, l, M)
    for k in range(0, T):
        f1 += useful_functions.E_log_norm(phi_m_k[:,k+J], data, mu_dp[k], nu_dp[k], lam_dp[k], psi_dp[k], p, l, M)

    # k = 0
    # m = 0
    #out = useful_functions.E_log_norm(data[m, :], mu_dp[k,:], nu_dp[k], lam_dp[k], psi_dp[k, :, :], p)
    print('f1 f2 done',f1,f2)
    #
    #
    # #f3 e f4
    f3=0
    f4=0
    for k in range(0,J):
        f3 += useful_functions.E_log_dens_norm_inv_wish(mu_mix[k,:],nu_mix[k],lam_mix[k],psi_mix[k,:,:],p,l)
    for k in range(0,T):
        f4 += useful_functions.E_log_dens_norm_inv_wish(mu_dp[k,:],nu_dp[k],lam_dp[k],psi_dp[k,:,:],p,l)
    print('f3 f4 done',f3,f4)

    #f5 #old version
    # f5=0
    # print(eta_k)
    # for k in range(0,J):
    #     for m in range(0,M):
    #         f5 += useful_functions.E_log_dens_dir(eta_k[k],J)*phi_m_k[m,k] #todo occhio che passa eta[k], ma poi
    #                                                                         #all internero di elogdens ancora eta[]
    # print('f5 done',f5)

    # Versione Jacopo
    #f5 = 0
    #for k in range(0, J):
    #    f5 += jnp.sum(phi_m_k[:, k]) * E_log_dens_dir_J(eta_k[k], eta_bar)#(eta-1)*(jdigamma(eta)-jdigamma(s_eta))
    f5 = jnp.sum(jnp.multiply(eta_k-1, diga_eta - diga_e_b))
    print('f5 done', f5)

    #f6 old
    # f6=0
    # for m in range(0,M):
    #     for k in range(0,T-1):
    #         s=0
    #         for h in range(0,k-J-1):
    #             s=float(s+jdgamma(b_k_beta[h])-jdgamma(a_k_beta[h]+b_k_beta[h]))
    #         f6 += float(phi_m_k[m,k]*(jdgamma(eta_k[0])-jdgamma(eta_bar)+jdgamma(a_k_beta[k-J])-jdgamma(a_k_beta[k-J]+b_k_beta[k-J])+s))

    # f6
    f6 = 0
    #for k in range(J, J + T - 1):
    #    s = 0
    #    for h in range(0, k - J - 1):
    #        s = float(s + jdgamma(b_k_beta[h]) - jdgamma(a_k_beta[h] + b_k_beta[h]))
    #    f6 += float(jnp.sum(phi_m_k[:, k]) * (jdgamma(eta_k[0]) - jdgamma(eta_bar) + jdgamma(a_k_beta[k - J]) - jdgamma(
    #        a_k_beta[k - J] + b_k_beta[k - J]) + s))
#
    #print('f6 done', f6)

    #
    parsum = jnp.cumsum(diga_b - diga_ab)

    for k in range(J, J + T - 1):
        f6 += jnp.sum(phi_m_k[:, k]) * (diga_eta[0] - diga_e_b + diga_a[k - J] - diga_ab[k-J] + parsum[k-J])
    print('f6 done', f6)

    #

    #f7
    f7=0
    #for k in range(0,J+1):
    #    f7 += float((a_dir_k[k]-1)*(jdgamma(eta_k[k])-jdgamma(eta_bar)))
    #
    #
    f7 = jnp.sum((a_dir_k-1)*(diga_eta-diga_e_b))
    #
    print('f7 done', f7)
    #f8
    #f8 = 0
    #for l in range(0,T-1):
    #    f8 += float((gamma-1)*(jdgamma(b_k_beta[l])-jdgamma(a_k_beta[l]+b_k_beta[l])))

    f8 = jnp.sum(jnp.multiply((gamma-1),(diga_b-diga_ab)))
    # E_log_p = f1+f2+f3+f4+f5+f6+f7+f8
    print('f8 done',f8)
    #val atteso log q

    #h1
    #h1=0
    #for m in range(0,M):
    #    #s=0
    #    #for h in range(0,J+T+1):
    #    #    s += phi_m_k[m,h]
    #    for k in range(0,J+T):
    #        h1 += phi_m_k[m,k]*jlog(phi_m_k[m,k])#-jlog(s)
    #print('h1 done', h1)

    #
    h1 = jnp.sum(jnp.multiply(phi_m_k,jlog(phi_m_k)))
    print('h1 done', h1)

    #h2
    h2=0
    q=0
    c=0
    #for j in range(0,J+1):
    #    h2 += float((eta_k[j]-1)*(jdgamma(eta_k[j])-jdgamma(eta_bar)))
    h2 = jnp.sum(jnp.multiply((eta_k-1),diga_eta-diga_e_b))
    #for j in range(0,J+1):
    #    q += float(eta_k[j])
    #    c += float(jgammaln(eta_k[j]))
    h2 += jgammaln(eta_bar)-jnp.sum(jgammaln(eta_k))
    print('h2 done', h2)

    #h3
    h3=0
   #for k in range(0,T-1):
   #    beta = (jnp.exp(jgammaln(a_k_beta[k])) * jnp.exp(jgammaln(b_k_beta[k])) / jnp.exp(jgammaln(a_k_beta[k] + b_k_beta[k])))
   #    h3 += float((a_k_beta[k]-1)*(jdgamma(a_k_beta[k])-jdgamma(b_k_beta[k]+a_k_beta[k])))
   #    h3 += float((b_k_beta[k]-1)*(jdgamma(b_k_beta[k])-jdgamma(a_k_beta[k]+b_k_beta[k]))-jlog(beta))
   #
    beta_AB = (jnp.exp(jgammaln(a_k_beta)) * jnp.exp(jgammaln(b_k_beta)) / jnp.exp(jgammaln(a_k_beta + b_k_beta)))
    h3 = jnp.sum(jnp.multiply(a_k_beta-1,diga_a - diga_ab)) + jnp.sum(jnp.multiply(b_k_beta-1,diga_b - diga_ab)) - jnp.sum(jnp.log(beta_AB))
    print('h3 done', h3)

    #h4 e h5
    h4=0
    for k in range(0,J):
        h4 += useful_functions.E_log_dens_norm_inv_wish(mu_mix[k,:],nu_mix[k],lam_mix[k],psi_mix[k,:,:],p,l)
    h5 = 0
    for k in range(0, T):
        h5 += useful_functions.E_log_dens_norm_inv_wish(mu_dp[k,:],nu_dp[k],lam_dp[k],psi_dp[k,:,:],p,l)

    E_log_q= h1+h2+h3+h4+h5
    print('h4 h5 done', h4,h5)

    # return E_log_p-E_log_q


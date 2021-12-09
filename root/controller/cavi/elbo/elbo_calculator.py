from VariationalBRAND.root.model.variational_parameters.variational_parameters import VariationalParameters
from VariationalBRAND.root.model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from VariationalBRAND.root.controller.cavi.utils import useful_functions
from jax import numpy as jnp
import jax.scipy.special.digamma as jdgamma
import jax.scipy.special.gamma as jgamma
import jax.numpy.log as jlog

def elbo_calculator(data,hyper: HyperparametersModel, var_param: VariationalParameters, p):
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

    mu_dp=nIW_DP_VAR.mu
    nu_dp=nIW_DP_VAR.nu
    lam_dp=nIW_DP_VAR.lamdA
    psi_dp=nIW_DP_VAR.phi


    #val atteso log p

    #f1 e f2
    f1=0
    f2=0
    for m in range(1,M+1):
        for k in range(1,J+1):
            f1 += phi_m_k[m,k]*useful_functions.E_log_norm(data[k,:],mu_mix[k,:],nu_mix[k],lam_mix[k],psi_mix[k,:,:],p)
        for k in range(J,J+T+1):
            f2 += phi_m_k[m,h]*useful_functions.E_log_norm(data[k,:],mu_dp[k,:],nu_dp[k],lam_dp[k],psi_dp[k,:,:],p)


    #f3 e f4
    f3=0
    f4=0
    for k in range(1,J+1):
        f3 += useful_functions.E_log_dens_norm_inv_wish(mu_mix[k,:],nu_mix[k],lam_mix[k],psi_mix[k,:,:],p)
    for k in range(J+1,J+T+1):
        f4 += useful_functions.E_log_dens_norm_inv_wish(mu_dp[k,:],nu_dp[k],lam_dp[k],psi_dp[k,:,:],p)

    #f5
    f5=0
    for k in range(1,J+1):
        for m in range(1,M+1):
            f5 += useful_functions.E_log_dens_dir(eta_k[k],J)*phi_m_k[m,k]

    #f6
    f6=0
    for m in range(1,M+1):
        for k in range(J+1,J+T+1):
            s=0
            for h in range(1,k-J-1+1):
                s=s+jdgamma(b_k_beta[h])-jdgamma(a_k_beta[h]+b_k_beta[h])
            f6 += phi_m_k[m,k]*(jdgamma(eta_k[k])-jdgamma(eta_bar)+jdgamma(a_k_beta[k-J])-jdgamma(a_k_beta[k-J]+b_k_beta[k-J])+s)

    #f7
    f7=0
    for k in range(0,J+1):
        f7 += (a_k_beta[k]-1)*(jdgamma(eta_k[k])-jdgamma(eta_bar))

    #f8
    f8 = 0
    for l in range(1,T+1):
        f7 += (gamma-1)*(jdgamma(b_k_beta[l])-jdgamma(a_k_beta[l]+b_k_beta[l]))

    E_log_p = f1+f2+f3+f4+f5+f6+f7+f8

    #val atteso log q

    #h1
    h1=0
    for m in range(1,M+1):
        s=0
        for h in range(0,J+T+1):
            s += phi_m_k[m,h]
        for k in range(1,J+T+1):
            h1 += phi_m_k[m,k]*jlog(phi_m_k[m,k])-jlog(s)

    #h2
    h2=0
    q=0
    c=0
    for j in range(0,J+1):
        h2 += (eta_k[j]-1)*(jdgamma(eta_k[j])-jdgamma(eta_bar))
    for j in range(0,J+1):
        q += eta_k[j]
        c += jlog(jgamma(eta_k[j]))
    h2 += jlog(jgamma(q))-c

    #h3
    h3=0
    for k in range(1,T+1):
        beta = (jgamma(a_k_beta[k]) * jgamma(b_k_beta[k]) / jgamma(a_k_beta[k] + b_k_beta[k]))
        h3 += (a_k_beta[k]-1)*(jdgamma(a_k_beta[k])-jdgamma(b_k_beta[k]+a_k_beta[k]))
        h3 += (b_k_beta[k]-1)*(jdgamma(b_k_beta[k])-jdgamma(a_k_beta[k]+b_k_beta[k]))-jlog(beta)

    #h4 e h5
    h4=0
    for k in range(1,J+1):
        h4 += useful_functions.E_log_dens_norm_inv_wish(mu_mix[k,:],nu_mix[k],lam_mix[k],phi_mix[k,:,:],p)
    h5 = 0
    for k in range(1, J+1):
        h5 += useful_functions.E_log_dens_norm_inv_wish(mu_dp[k,:],nu_dp[k],lam_dp[k],phi_dp[k,:,:],p)

    E_log_q= h1+h2+h3+h4+h5

    return E_log_p-E_log_q



import jax
import jax.numpy as jnp
import jax.scipy as js
from jax.numpy.linalg import det as jdet, pinv as jinv
from jax.scipy.special import digamma as jdigamma

# val atteso log beta
def E_log_beta_unjitted(a,b):
    # a,b coeffs logbeta
    return jdigamma(a)-jdigamma(a+b)
E_log_beta = jax.jit(E_log_beta_unjitted)
# val atteso log densità normal inverse wishart

#def E_log_dens_norm_inv_wish_q(mu,nu,lam,psi,p,l):
#    # p dim of mu
#    ret = +jnp.log(jdet(psi))
#    ret = ret -jnp.sum(jdigamma((nu - l) / 2))
#    ret = ret - p*jnp.log(lam)
#
#    ret = -0.5*ret
#    #em = jnp.exp(js.special.multigammaln(nu/2,p))
#    #ret = ret + jnp.log((jdet(psi)**(nu/2))/((2**(nu*p/2))*em))
#    #
#    ret = ret + nu*jnp.log(jdet(psi))/2 - (nu*p/2)*jnp.log(2) - js.special.multigammaln(nu/2,p)
#
#
#    brut = p*jnp.log(2) - jnp.log(jdet(psi))
#
#    brut = brut + jnp.sum(jdigamma((nu - l) / 2))
#    ret = ret + brut*(nu+p+1)/2
#    ret = ret - p*nu/2
#    return ret

def E_log_dens_norm_inv_wish_q(mu,nu,lam,psi,p,l):
    ret = callable_1_q(mu,nu,lam,psi,p,l)
    ret = ret - js.special.multigammaln(nu/2,p)

    ret = ret + callable_2_q(mu,nu,lam,psi,p,l)
    return ret

def callable_1_q_unjitted(mu,nu,lam,psi,p,l):
    # p dim of mu
    ret = +jnp.log(jdet(psi))
    ret = ret -jnp.sum(jdigamma((nu - l) / 2))
    ret = ret - p*jnp.log(lam)

    ret = -0.5*ret
    #em = jnp.exp(js.special.multigammaln(nu/2,p))
    #ret = ret + jnp.log((jdet(psi)**(nu/2))/((2**(nu*p/2))*em))
    #
    ret = ret + nu*jnp.log(jdet(psi))/2 - (nu*p/2)*jnp.log(2)
    return ret

callable_1_q = jax.jit(callable_1_q_unjitted)

def callable_2_q_unjitted(mu,nu,lam,psi,p,l):
    brut = p*jnp.log(2) - jnp.log(jdet(psi))

    brut = brut + jnp.sum(jdigamma((nu - l) / 2))
    ret = brut*(nu+p+1)/2
    ret = ret - p*nu/2
    return ret
callable_2_q = jax.jit(callable_2_q_unjitted)

def E_log_dens_norm_inv_wish_p(mu_var,nu_var,lam_var,psi_var,mu_0,nu_0,lam_0,psi_0,p,l):
    # p dim of mu
    mu_var = jnp.reshape(mu_var, p)
    mu_0 = jnp.reshape(mu_0, p)
    psi_0 = jnp.reshape(psi_0, (p,p))
    nu_0 = float(nu_0)
    lam_0 = float(lam_0)
    ret = callable_p(mu_var,nu_var,lam_var,psi_var,mu_0,nu_0,lam_0,psi_0,p,l)
    return ret


def callable_p_unjitted(mu_var,nu_var,lam_var,psi_var,mu_0,nu_0,lam_0,psi_0,p,l):
    # p dim of mu
    ret = jnp.log(jdet(psi_var))
    ret = ret -jnp.sum(jdigamma((nu_var - l) / 2))
    ret = ret - p*jnp.log(lam_0) + p*lam_0/lam_var
    ret = ret + lam_0*nu_var*((mu_var-mu_0).T @ jinv(psi_var) @ (mu_var-mu_0))

    ret = -0.5*ret

    #em = jnp.exp(js.special.multigammaln(nu/2,p))
    #ret = ret + jnp.log((jdet(psi)**(nu/2))/((2**(nu*p/2))*em))
    #


    brut = p*jnp.log(2) - jnp.log(jdet(psi_var))

    brut = brut + jnp.sum(jdigamma((nu_var - l) / 2))
    ret = ret + brut*(nu_0+p+1)/2
    ret = ret - 0.5*nu_var*jnp.trace(psi_0@jinv(psi_var))
    return ret
callable_p = jax.jit(callable_p_unjitted)

# val atteso log densità dirichlet
# def E_log_dens_dir(eta,J):
#     #print(eta)
#     s_eta = jnp.sum(eta)
#     ret = 0
#     for j in range(0,J+1):
#         ret = ret + \
#               (eta[j]-1)*\
#               (jdigamma(eta[j])
#                -jdigamma(s_eta))
#     return ret

#Versione Jacopo
#def E_log_dens_dir_unjitted(eta : float ,s_eta : float,):
#   ret = (eta-1)*(jdigamma(eta)-jdigamma(s_eta))
#   return ret
#E_log_dens_dir_J = jax.jit(E_log_dens_dir_unjitted)

# val atteso log densità beta 
#def E_log_dens_beta_unjitted(a:float,b:float):
#    b = jnp.exp(jgamma(a)*jgamma(b)/jgamma(a+b))
#    return (a-1)*E_log_beta(a,b) + (b-1)*E_log_beta(b,a) - jnp.log(b)
#E_log_dens_beta = jax.jit(E_log_dens_beta_unjitted)

#val atteso log normale dati
#def E_log_norm(data,mu,nu,lam,psi,p):
#    mu = jnp.reshape(mu, p)
#    data = jnp.reshape(data, p)
#    psi = jnp.reshape(psi, (p,p))
#
#    ret = -jnp.log(jdet(jinv(psi)))
#    for i in range(1,p+1):
#        ret = ret - jdigamma((nu-i+1)/2)
#    ret = ret+p/lam
#    ret = ret+ nu*jnp.dot(data-mu,jnp.dot(jinv(psi),data-mu))
#    ret = -0.5*ret
#    #print(ret)
#    return ret

# -1 / 2 * (-jnp.sum(jdgamma((nu_mix[k] - l) / 2)) + jnp.log(jdet(psi_mix[k, :, :]))
#                       + p / lam_mix[k] + nu_mix[k] * jnp.sum(jnp.diag(((data - mu_mix[k, :]) @ jinv(psi_mix[k, :, :]) @
#                                                                           (data - mu_mix[k, :]).T))))

#def E_log_norm_unjitted(data,mu,nu,lam,psi,p, l):
#    #mu = jnp.reshape(mu, p)
#    #data = jnp.reshape(data, p)
#    #psi = jnp.reshape(psi, (p,p))
#
#    ret = -1/2*(-jnp.sum(jdigamma((nu-l)/2)) + jnp.log(jdet(psi))
#                      + p/lam + nu*((data - mu).T @ jinv(psi) @(data - mu)))
#
#    return ret
#E_log_norm = jax.jit(E_log_norm_unjitted)
#


#def E_log_norm_old(phi, data,mu,nu,lam,psi,p, l,M):
#    mu = jnp.reshape(mu, p)
#    data = jnp.reshape(data, (M,p))
#    psi = jnp.reshape(psi, (p,p))
#    phi = jnp.reshape(phi, M)
#
#    ret = jnp.multiply(-jnp.sum(jdigamma((nu-l)/2)) + jnp.log(jdet(psi)) + p/lam,jnp.ones(M))
#    ret+= nu*(jnp.diag((data - mu) @ jinv(psi) @(data - mu).T))
#    ret = -0.5*ret
#    ret = jnp.dot(phi,ret)
#    return ret
#


def E_log_norm(phi, data,mu,nu,lam,psi,p, l,M):
    mu = jnp.reshape(mu, p)
    data = jnp.reshape(data, (M,p))
    psi = jnp.reshape(psi, (p,p))
    phi = jnp.reshape(phi, M)

    detpsi = jdet(psi)

    if detpsi != 0:
        ret = jnp.multiply(-jnp.sum(jdigamma((nu-l)/2)) + jnp.log(detpsi) + p/lam,jnp.ones(M))
    else:
        ret = jnp.multiply(-jnp.sum(jdigamma((nu - l) / 2)) + p / lam, jnp.ones(M))

    ret = callable_norm(ret,phi, data,mu,nu,lam,psi,M)
    
    return ret


def callable_norm_unjitted(val,phi, data,mu,nu,lam,psi,M):
    ret = val + nu*(jnp.diag((data - mu) @ jinv(psi) @ (data - mu).T))
    ret = -0.5*ret
    ret = jnp.dot(phi, ret)
    return ret


callable_norm = jax.jit(callable_norm_unjitted)


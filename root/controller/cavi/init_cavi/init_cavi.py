from jax import numpy as jnp
from root.model.NIW import NIW
from root.model.user_input_model import UserInputModel
from root.model.variational_parameters import VariationalParameters


def init_cavi(user_input_parameters : UserInputModel):
    return VariationalParameters(
        phi_m_k = jnp.array(user_input_parameters.Phi_m_k),
        eta_k = jnp.reshape(jnp.array(user_input_parameters.eta_k), user_input_parameters.J + 1),
        a_k_beta = jnp.array(user_input_parameters.a_k_beta),
        b_k_beta = jnp.array(user_input_parameters.b_k_beta),

        nIW_VAR=NIW(
            mu = jnp.concatenate((jnp.array(user_input_parameters.mu_VAR_MIX),
                                 jnp.array(user_input_parameters.mu_var_DP)),
                                 axis=0),
            nu = jnp.concatenate((jnp.array(user_input_parameters.nu_VAR_MIX),
                                  jnp.array(user_input_parameters.nu_var_DP)),
                                 axis=0),
            lambdA = jnp.concatenate((jnp.array(user_input_parameters.lambda_VAR_MIX),
                                     jnp.array(user_input_parameters.lambda_var_DP)),
                                     axis=0),
            phi = jnp.concatenate((jnp.array(user_input_parameters.PHI_VAR_MIX),
                                   jnp.array(user_input_parameters.PHI_var_DP)),
                                  axis=0)
        )
    )
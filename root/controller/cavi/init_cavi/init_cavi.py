from model.user_input import UserInput
from model.variational_parameters.variational_parameters import VariationalParameters
from jax import numpy as jnp
from root.model.hyperparameters_model.NIW import NIW

def init_cavi(user_input_parameters : UserInput):
    return VariationalParameters(
        #phi_m_k=phi_m_k_temp,
        phi_m_k = user_input_parameters.Phi_m_k,

        #eta_k=hyperparameters_model.a_dir_k,
        eta_k=user_input_parameters.eta_k,

        #a_k_beta=jnp.ones(T - 1),
        a_k_beta = jnp.array(user_input_parameters.a_k_beta),

        #b_k_beta=jnp.multiply(jnp.ones(T - 1), hyperparameters_model.gamma),
        b_k_beta = jnp.array(user_input_parameters.b_k_beta),

        nIW_DP_VAR=NIW(
            #mu=jnp.repeat(hyperparameters_model.nIW_DP_0.mu_0_DP, repeats=T, axis=0),
            mu = jnp.array(user_input_parameters.mu_var_DP),

            # nu=jnp.multiply(jnp.ones(T), hyperparameters_model.nIW_DP_0.nu_0_DP),
            nu = jnp.array(user_input_parameters.nu_var_DP),

            # lambdA=jnp.multiply(jnp.ones(T), hyperparameters_model.nIW_DP_0.lambda_0_DP),
            lambdA = jnp.array(user_input_parameters.lambda_var_DP),

            # phi=jnp.repeat(hyperparameters_model.nIW_DP_0.phi_0_dp, repeats=T, axis=0)),
            phi = jnp.array(user_input_parameters.PHI_var_DP)
        ),

        # nIW_MIX_VAR=hyperparameters_model.nIW_MIX_0,
        nIW_MIX_VAR = NIW(
            mu=user_input_parameters.mu_VAR_MIX,
            nu=user_input_parameters.nu_VAR_MIX,
            lambdA=user_input_parameters.lambda_VAR_MIX,
            phi=user_input_parameters.PHI_VAR_MIX,
        )

    )
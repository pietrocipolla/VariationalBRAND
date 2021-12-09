from model.user_input import UserInput
from root.model.hyperparameters_model.NIW import NIW
from root.model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from jax import numpy as jnp

def set_hyperparameters(user_input_parameters: UserInput):
    return HyperparametersModel(
        #gamma=5,
        gamma=user_input_parameters.gamma,

        #a_dir_k=jnp.ones(3 + 1),
        a_dir_k=jnp.array(user_input_parameters.a_dir_k),

        nIW_DP_0=NIW(
            # mu=jnp.ones(2)[None, :],  # così che sia comunque della forma n_elems x p
            mu=jnp.array(user_input_parameters.mu_0_DP),
            # nu=jnp.array([2]),
            nu = jnp.array(user_input_parameters.nu_0_DP),
            # lambdA=jnp.array([1]),
            lambdA = jnp.array(user_input_parameters.lambda_0_DP),
            # phi=jnp.identity(2)[None, :]
            phi = jnp.array(user_input_parameters.PHI_0_DP)
            # vettore (p) componenti, con p = 2
        ),

        nIW_MIX_0=NIW(
            # mu=mu_0_MIX,
            mu = jnp.array(user_input_parameters.mu_0_MIX),
            # nu=jnp.multiply(jnp.ones(3), 2),
            nu=jnp.array(user_input_parameters.nu_0_MIX),
            # lambdA=jnp.ones(3),
            lambdA = jnp.array(user_input_parameters.lambda_0_MIX)
            # phi=phi_0_MIX,  # varianza #todo check coerenza robust estimaros
        )  # TODO trovare inizializzazione più furba
    )
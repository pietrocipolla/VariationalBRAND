from model.user_input import UserInputModel
from root.model.hyperparameters_model.NIW import NIW
from root.model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from jax import numpy as jnp

def set_hyperparameters(user_input_parameters: UserInputModel):
    return HyperparametersModel(
        gamma=user_input_parameters.gamma,

        a_dir_k=jnp.array(user_input_parameters.a_dir_k),

        nIW_DP_0=NIW(
            mu=jnp.array(user_input_parameters.mu_0_DP),
            nu = jnp.array(user_input_parameters.nu_0_DP),
            lambdA = jnp.array(user_input_parameters.lambda_0_DP),
            phi = jnp.array(user_input_parameters.PHI_0_DP)
        ),

        nIW_MIX_0=NIW(
            mu = jnp.array(user_input_parameters.mu_0_MIX),
            nu=jnp.array(user_input_parameters.nu_0_MIX),
            lambdA = jnp.array(user_input_parameters.lambda_0_MIX),
            phi = jnp.array(user_input_parameters.PHI_0_MIX)
        )
    )
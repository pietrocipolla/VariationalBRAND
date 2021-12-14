from jax import numpy as jnp
from root.model.NIW import NIW
from root.model.hyperparameters_model import HyperparametersModel
from root.model.user_input_model import UserInputModel
#convert numpy array to jax array https://github.com/google/jax/issues/1961
#jax_array = jnp.array(numpy_array)

def set_hyperparameters(user_input_parameters: UserInputModel, Y):
    return HyperparametersModel(
        J=user_input_parameters.J,
        T=user_input_parameters.T,
        M=Y.shape[0],
        p=Y.shape[1],  # number of data coordinates
        n_iter=user_input_parameters.n_iter,
        tol=user_input_parameters.tol,

        gamma = user_input_parameters.gamma,

        a_dir_k = jnp.array(user_input_parameters.a_dir_k),

        nIW_DP_0 = NIW(
            mu = jnp.reshape(jnp.array(user_input_parameters.mu_0_DP),(1,Y.shape[1])),
            nu = jnp.array(user_input_parameters.nu_0_DP),
            lambdA = jnp.array(user_input_parameters.lambda_0_DP),
            phi = jnp.array(user_input_parameters.PHI_0_DP)
        ),

        nIW_MIX_0 = NIW(
            mu = jnp.array(user_input_parameters.mu_0_MIX),
            nu = jnp.array(user_input_parameters.nu_0_MIX),
            lambdA = jnp.array(user_input_parameters.lambda_0_MIX),
            phi = jnp.array(user_input_parameters.PHI_0_MIX)
        ),
    )

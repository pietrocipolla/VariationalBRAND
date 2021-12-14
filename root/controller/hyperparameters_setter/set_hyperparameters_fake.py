from jax import numpy as jnp
from root.model.NIW import NIW
from root.model.hyperparameters_model import HyperparametersModel


def set_hyperparameters_fake(hyperparameters_model : HyperparametersModel, mu_0_MIX, phi_0_MIX): #todo nel caso agigugnere check dimensioni array
    return HyperparametersModel(
        gamma=5,
        # gamma -> parametro dello Stick Breaking -> scalare
        # iperparametro tra 1 e 50 tipo oppure buttarci su una distribuzione e una prior

        a_dir_k=jnp.ones(3 + 1),
        # a_dir_k -> vettore delle componenti della Dirichlet -> vettore di (J+1) componenti
        # J = num_classes_learning

        nIW_DP_0=NIW(
            mu=jnp.ones(2)[None, :],  # così che sia comunque della forma n_elems x p
            nu=jnp.array([2]),
            lambdA=jnp.array([1]),
            phi=jnp.identity(2)[None, :]
            # vettore (p) componenti, con p = 2
        ),

        nIW_MIX_0=NIW(
            mu=mu_0_MIX,
            nu=jnp.multiply(jnp.ones(3), 2),
            lambdA=jnp.ones(3),
            phi=phi_0_MIX,  # varianza #todo check coerenza robust estimaros
        )  # TODO trovare inizializzazione più furba
    )


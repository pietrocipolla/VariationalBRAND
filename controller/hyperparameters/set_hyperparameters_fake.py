from model.hyperparameters_model.NIW import NIW
from model.hyperparameters_model.NIW_DP_0.NIW_DP_0_parameters.Lambda_0_DP import Lambda_0_DP
from model.hyperparameters_model.NIW_DP_0.NIW_DP_0_parameters.Mu_0_DP import Mu_0_DP
from model.hyperparameters_model.NIW_DP_0.NIW_DP_0_parameters.Nu_0_DP import Nu_0_DP
from model.hyperparameters_model.NIW_DP_0.NIW_DP_0_parameters.PHI_0_DP import PHI_0_DP
from model.hyperparameters_model.NIW_MIX_0.NIW_MIX_0 import NIW_MIX_0
from model.hyperparameters_model.a_dir_k import A_dir_k
from model.hyperparameters_model.gamma import Gamma
from model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from model.hyperparameters_model.NIW_DP_0.NIW_DP_0 import NIW_DP_0
from model.hyperparameters_model.NIW_MIX_0.NIW_MIX_0_parameters.Mu_0_MIX import Mu_0_MIX
from model.hyperparameters_model.NIW_MIX_0.NIW_MIX_0_parameters.Nu_0_MIX import Nu_0_MIX
from model.hyperparameters_model.NIW_MIX_0.NIW_MIX_0_parameters.Lambda_0_MIX import Lambda_0_MIX
from model.hyperparameters_model.NIW_MIX_0.NIW_MIX_0_parameters.PHI_0_MIX import PHI_0_MIX
from jax import numpy as jnp

def set_hyperparameters_fake(ask_hyperparameters_from_user_input,
                             num_classes, num_classes_learning, num_classes_test, robust_mean, n_samples):
    return HyperparametersModel(
        gamma =  5,
        # gamma -> parametro dello Stick Breaking -> scalare
        # iperparametro tra 1 e 50 tipo oppure buttarci su una distribuzione e una prior

        a_dir_k = jnp.ones(3+1),
        #a_dir_k -> vettore delle componenti della Dirichlet -> vettore di (J+1) componenti
        # J = num_classes_learning

        nIW_DP_0 = NIW(

        )
        NIW_DP_0(
            Mu_0_DP(
                jnp.ones(2)
                #vettore (p) componenti, con p = 2
            ),
            Nu_0_DP(
                2
                #p = 2 numero di componenti del vettore di y
            ),
            Lambda_0_DP(
                1
                #-> scalare
            ),
            PHI_0_DP(
                jnp.identity(2)
                # matrice identità p x p con p = 2
            )

        ),
        NIW_MIX_0(
            Mu_0_MIX(
                jnp.array([[2, 3], [4, 5], [1, 2]])
            ),
            Nu_0_MIX(
                jnp.ones(3)
                #J = 3
            ),
            Lambda_0_MIX(
                jnp.ones(3)
                # J = 3
            ),
            PHI_0_MIX(
                jnp.array([[[2,3],[4,5],[1,2]],
                          [[2,3],[4,5],[1,2]]])
            )
        )
    )


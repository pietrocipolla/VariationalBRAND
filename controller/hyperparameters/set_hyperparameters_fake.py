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

def set_hyperparameters_fake(mu_0_MIX, phi_0_MIX): #todo nel caso agigugnere check dimensioni array
    return HyperparametersModel(
        gamma =  5,
        # gamma -> parametro dello Stick Breaking -> scalare
        # iperparametro tra 1 e 50 tipo oppure buttarci su una distribuzione e una prior

        a_dir_k = jnp.ones(3+1),
        #a_dir_k -> vettore delle componenti della Dirichlet -> vettore di (J+1) componenti
        # J = num_classes_learning

        nIW_DP_0 = NIW(
            mu_0 = jnp.oneseyeones(2)[None, :], #così che sia comunque della forma n_elems x p
            nu_0 = 2,
            lambda_0 = 1,
            phi_0 = jnp.identity(2)
            # vettore (p) componenti, con p = 2
        ),

        nIW_MIX_0 = NIW(
            mu_0 = mu_0_MIX,
            nu_0 =  jnp.array([2, 2, 2]),
            lambda_0= jnp.ones(3),
            phi_0 = phi_0_MIX, #varianza
        ) #TODO trovare inizializzazione più furba

    )


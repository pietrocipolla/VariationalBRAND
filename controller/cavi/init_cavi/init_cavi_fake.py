from jax import numpy as jnp

from model.hyperparameters_model.hyperparameters_model import HyperparametersModel
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR import NIW_DP_VAR
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters.Lambda_VAR_DP import Lambda_VAR_DP
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters.Mu_VAR_DP import Mu_VAR_DP
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters.Nu_VAR_DP import Nu_VAR_DP
from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters.PHI_VAR_DP import PHI_VAR_DP
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR import NIW_MIX_VAR
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR_parameters.Lambda_VAR_MIX import Lambda_VAR_MIX
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR_parameters.Mu_VAR_MIX import Mu_VAR_MIX
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR_parameters.Nu_VAR_MIX import Nu_VAR_MIX
from model.variational_parameters.NIW_MIX_VAR.NIW_MIX_VAR_parameters.PHI_VAR_MIX import PHI_VAR_MIX
from model.variational_parameters.variational_parameters import VariationalParameters
from model.variational_parameters.a_k_beta import A_k_beta
from model.variational_parameters.b_k_beta import B_k_beta
from model.variational_parameters.eta_k import Eta_k
from model.variational_parameters.phi_m_k import Phi_m_k

def init_cavi_fake(hyperparameters_model: HyperparametersModel):
    return VariationalParameters(
        Phi_m_k(jnp.array([[0.1, 0.3, 0.2, 0.0, 0.5],
                                              [0.1, 0.3, 0.2, 0, 0.5]])),
        # todo
        # RICORDARSI DI NORMALIZZARE A SOMMA 1 DOPO AVERLE AGGIORNATE O TUTTO VA A T*OIE

        Eta_k(jnp.ones(3 + 1)),
        # J = 3

        # V_i ~ Beta(a_k[i-1], b_k[i-1])
        A_k_beta(2 - 1),
        # T = 2

        B_k_beta(2 - 1),
        # T = 2

        NIW_DP_VAR(
            Mu_VAR_DP(
                jnp.array([[2,3],[4,5]])
                # matrice (Txp) -> riga per riga ci sono le medie delle componenti della misturaDP
                # (mu_var_DP[i,:] -> media della (i+1)-esima NIW della misturaDP)
            ),
            Nu_VAR_DP(
                jnp.ones(2)
                # T = 2
                # vettore (T) componenti
                # (nu_var_MIX[i] = nu_var della (i+1) esima NIW della misturaDP)
            ),
            Lambda_VAR_DP(
                jnp.ones(2)
                # T = 2
                # (lambda_var_DP[i] = lambda_var della (i+1) esima NIW della misturaDP)
            ),
            PHI_VAR_DP(
                jnp.array([ [[2,3],[4,5]],[[2,3],[4,5]]])
                # T = 2
                # p = 2
                # -> vettore di matrici
                # (Txpxp), sostanzialmente un ndarray -> PHI_var_DP[i,:,:] = Matrice PHI della (i+1)esima NIW della misturaDP
            )

        ),
        NIW_MIX_VAR(
            Mu_VAR_MIX(
                # J = 3
                # p = 2
                jnp.array([[2,3],[4,5],[1,2]])
            ),
            Nu_VAR_MIX(
                jnp.ones(3)
                # J = 3
            ),
            Lambda_VAR_MIX(
                jnp.ones(3)
                # J = 3
            ),
            PHI_VAR_MIX(
                jnp.array([[[2, 3], [4, 5], [1, 2]],
                           [[2, 3], [4, 5], [1, 2]]])
            )
        )
    )

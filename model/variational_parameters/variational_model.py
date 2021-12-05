from model.variational_parameters.NIW_DP_VAR import NIW_DP_VAR
from model.variational_parameters.NIW_MIX_VAR import NIW_MIX_VAR
from model.variational_parameters.a_k_beta import A_k_beta
from model.variational_parameters.b_k_beta import B_k_beta
from model.variational_parameters.eta_k import Eta_k
from model.variational_parameters.phi_m_k import Phi_m_k


class VariationalParameters:
    def __init__(self,  phi_m_k: Phi_m_k,  eta_k: Eta_k, a_k_beta: A_k_beta, b_k_beta: B_k_beta,
                 nIW_DP_VAR : NIW_DP_VAR, nIW_MIX_VAR: NIW_MIX_VAR):
        self.phi_m_k = phi_m_k
        self.eta_k = eta_k
        self.a_k_beta = a_k_beta
        self.b_k_beta = b_k_beta
        self.nIW_DP_VAR = nIW_DP_VAR
        self.nIW_MIX_VAR = nIW_MIX_VAR

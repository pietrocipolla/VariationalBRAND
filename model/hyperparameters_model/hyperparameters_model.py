from model.hyperparameters_model.NIW_DP_0 import NIW_DP_0
from model.hyperparameters_model.NIW_MIX_0 import NIW_MIX_0
from model.hyperparameters_model.a_dir_k import A_dir_k
from model.hyperparameters_model.gamma import Gamma

class HyperparametersModel:
    def __init__(self, gamma: Gamma, a_dir_k : A_dir_k, nIW_DP_0 : NIW_DP_0, nIW_MIX_0: NIW_MIX_0):
        self.gamma = gamma
        self.a_dir_k = a_dir_k
        self.nIW_DP_0 = nIW_DP_0
        self.nIW_MIX_0 = nIW_MIX_0



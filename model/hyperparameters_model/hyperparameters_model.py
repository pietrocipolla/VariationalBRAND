from model.hyperparameters_model import NIW
from jax import numpy as jnp

class HyperparametersModel:
    def __init__(self, gamma: int, a_dir_k : jnp.array, nIW_MIX_0 : NIW, nIW_DP_0: NIW, J : int, T : int):
        #Per inizializzare distribuzione Beta(1,gamma) per Stick Breaking
        self.gamma = gamma
        self.a_dir_k = a_dir_k
        self.nIW_DP_0 = nIW_DP_0
        self.nIW_MIX_0 = nIW_MIX_0



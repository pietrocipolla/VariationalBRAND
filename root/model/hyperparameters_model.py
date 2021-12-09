from model.NIW import NIW
from jax import numpy as jnp

class HyperparametersModel:
    def __init__(self, gamma: int, a_dir_k : jnp.array, nIW_MIX_0 : NIW, nIW_DP_0: NIW, J : int, T : int, M : int):
        #Per inizializzare distribuzione Beta(1,gamma) per Stick Breaking
        self.gamma = gamma
        #Per inizializzare distribuzione di Dirichlet (guardare Ghirri et al. 2021)
        self.a_dir_k = a_dir_k
        # Per inizializzare Normal inverse-Wishart dei dati del Dirichlet Process
        self.nIW_DP_0 : NIW = nIW_DP_0
        # Per inizializzare Normal inverse-Wishart dei dati del test set
        self.nIW_MIX_0 : NIW = nIW_MIX_0
        #Numero di classi nel training set
        self.J = J
        #Numero di classi massime nel Dirichlet Process
        self.T = T
        # Numero totale di dati (size data frame ossia numero di righe)
        self.M = M



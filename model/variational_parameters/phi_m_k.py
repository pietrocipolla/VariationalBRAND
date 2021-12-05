from jax import numpy as jnp

class Phi_m_k:
    #> parametri delle multinomiali -> matrice (Mx(J+T))
    # -> phi_m_k[i:] = parametri della (i+1)esima multinomiale
    # -> phi_m_k[i,j] = prob che y_i sia nel cluster j

    #todo #RICORDARSI DI NORMALIZZARE A SOMMA 1 DOPO AVERLE AGGIORNATE O TUTTO VA A *ROIE

    def __init__(self, Phi_m_k: jnp.array):
        self.Phi_m_k = jnp.array(Phi_m_k)

# # example matrice (Mx(J+T)
# M = 2
# J = 3
# T = 2
# #J+T = 5
#
# user_input = jnp.array([[0.1,0.3, 0.2,0.0,0.5],
#                            [0.1,0.3, 0.2,0,0.5]])
#
# hello = Phi_m_k(user_input)
# print(hello.Phi_m_k_matrix)
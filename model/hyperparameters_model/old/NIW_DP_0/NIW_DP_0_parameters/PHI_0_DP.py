from jax import numpy as jnp

class PHI_0_DP:
    #-> Matrice (pxp)
    def __init__(self, PHI_0_DP: jnp.array):
        self.PHI_0_DP = PHI_0_DP

#example
# p = 2
# user_input = [[1.0, 1.0],
#                [1.0, 1.0]]
# hello = PHI_0_DP(user_input)
# print(hello.PHI_0_DP)
#
# user_input = jnp.identity(p)
# hello = PHI_0_DP(user_input)
# print(hello.PHI_0_DP)
from jax import numpy as jnp

class PHI_VAR_DP:
    #vettore di matrici (Txpxp), sostanzialmente un ndarray
    # -> PHI_var_DP[i,:,:] = Matrice PHI della (i+1)esima NIW della misturaDP
    def __init__(self, PHI_VAR_DP: jnp.ndarray):
        self.PHI_VAR_DP = jnp.array(PHI_VAR_DP)

# # example
# T = 2
# p = 2
# user_input = jnp.array([
#                         [[2,3],[4,5]],
#                         [[2,3],[4,5]]])
#
# hello = PHI_VAR_DP(user_input)
# print(hello.PHI_VAR_DP)

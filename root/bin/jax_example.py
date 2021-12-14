#from https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/jax_for_the_impatient.ipynb#scrollTo=L2HKiLTNJ4Eh
import jax
from jax import numpy as jnp, random
import numpy as np # We import the standard NumPy library

m = jnp.ones((4,4)) # We're generating one 4 by 4 matrix filled with ones.
n = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]]) # An explicit 2 by 4 array
m

jnp.dot(n, m).block_until_ready() # Note: yields the same result as np.dot(m)
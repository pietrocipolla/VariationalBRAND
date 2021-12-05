#https://docs.python.org/3/library/typing.html
#https://www.w3schools.com/python/python_datatypes.asp

import jax
from jax import numpy as jnp, random
import numpy as np # We import the standard NumPy library

class HyperParameters:
#when the project is stabilized, evaluate to pass from parameters list to class objects
#example
    x = 5

class gamma:
# gamma -> parametro dello Stick Breaking -> scalare

    def __init__(self, gamma : int):
        self.gamma = gamma

#example
# user_input = 1
# hello = gamma(user_input)
# print(hello.gamma)
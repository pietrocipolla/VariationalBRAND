import numpy
import numpy as np
import random
import matplotlib.pyplot as plt

def test_init(mu_0_DP, PHI_0_DP):
    mean = mu_0_DP
    cov = PHI_0_DP
    x, y = np.random.multivariate_normal(mean, cov, 1).T


    return numpy.asarray([float(x),float(y)])

    # plt.plot(x, y, 'x')
    #
    # plt.axis('equal')
    #
    # plt.show()

# p = 2
# T = 20
# mu_0_DP = np.ones(p)
# PHI_0_DP = np.multiply(np.identity(p), 10)
# out = test_init(mu_0_DP, PHI_0_DP)
# print(out)

def test_mu_var_DP_init():
    p = 2
    T = 10
    mu_0_DP = np.ones(p)
    #mu_0_DP = np.zeros(p)
    PHI_0_DP = np.multiply(np.identity(p), 10)

    gauss_realizations = []
    for i in range(T):
        gauss_realizations.append(test_init(mu_0_DP, PHI_0_DP))

    #mu_var_DP = np.tile(mu_0_DP, (T, 1))

    x =  gauss_realizations
    y = numpy.array([numpy.array(xi) for xi in x])
    mu_var_DP = y
    print(mu_var_DP)
    return mu_var_DP

test_mu_var_DP_init()
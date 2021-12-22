import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# FEMALE, MALE = 0, 1
# dt = np.dtype([('mass', 'f8'), ('height', 'f8'), ('gender', 'i2')])
# data = np.loadtxt('body.dat.txt', usecols=(22,23,24), dtype=dt)
#
# fig, ax = plt.subplots()

def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)

def plot_cov_ellipse(mu,cov):
    ax = plt.gca()
    e = get_cov_ellipse(cov, mu, 0.2, alpha=0.4)
    ax.add_artist(e)

    # ax.set_xlim(-15, 15)
    # ax.set_ylim(-15, 15)
    # ax.set_xlabel('Height /cm')
    # ax.set_ylabel('Mass /kg')
    # ax.legend(loc='upper left', scatterpoints=1)
    # plt.show()

from unittest import TestCase

from controller.plotter.generate_elbo_plot import generate_elbo_plot


class Test(TestCase):
    def test_generate_elbo_plot(self):
        elbo_values = [5,4,5,3,2,1,0,0,0]
        generate_elbo_plot(elbo_values)

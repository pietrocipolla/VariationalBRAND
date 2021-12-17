from unittest import TestCase
from root.controller.sample_data_handler.data_generator import generate_some_data_example
from numpy import savetxt


class Test(TestCase):
    def test_generate_some_data_example(self):
        X = generate_some_data_example()
        # print(X.shape)
        #print(X.shape[1])
        self.assertEqual(X.shape, (750, 2))

        savetxt('data.csv', X, delimiter=',')

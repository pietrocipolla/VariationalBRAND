from unittest import TestCase
from root.controller.sample_data_handler.data_generator import generate_some_data_example


class Test(TestCase):
    def test_generate_some_data_example(self):
        X = generate_some_data_example()
        # print(X.shape)
        self.assertEqual(X.shape, (500, 2))

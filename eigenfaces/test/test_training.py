import unittest
import numpy as np
import numpy.testing as npt
import src.training as tr


class Test(unittest.TestCase):

    def test_divide_dataset_into_train_and_test_sets(self):
        dataset = np.arange(200).reshape(20, 10)

        (X_train, X_test) = tr.divide_dataset_into_train_test(dataset, 0.9)

        actual_training_set_size = len(X_train)
        expected_training_set_size = 18
        actual_test_set_size = len(X_test)
        expected_test_set_size = 2

        npt.assert_equal(actual_training_set_size, expected_training_set_size)
        npt.assert_equal(actual_test_set_size, expected_test_set_size)

    def test_mean_of_training_set(self):
        X_train = np.array([[2, 5, 6, 8],
                            [4, 4, 3, 6],
                            [6, 5, 6, 1],
                            [2, 4, 5, 2]])

        actual = tr.mean(X_train)
        expected = np.array([3.5, 4.5, 5, 4.25])

        npt.assert_equal(actual, expected)

    def test_subtract_mean_from_training_set(self):
        X_train = np.array([[2, 5, 6, 8],
                            [4, 4, 3, 6],
                            [6, 5, 6, 1],
                            [2, 4, 5, 2]])

        actual = tr.zero_mean(X_train)

        expected = [[-1.5, 0.5, 1., 3.75],
                    [ 0.5,-0.5,-2., 1.75],
                    [ 2.5, 0.5, 1.,-3.25],
                    [-1.5,-0.5, 0.,-2.25]]

        npt.assert_equal(actual, expected)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_divide_dataset_into_train_and_test_sets']
    unittest.main()

import unittest
import numpy as np
import numpy.testing as npt
from math import sqrt
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

        X_test = np.array([[1, 1, 1, 1],
                           [2, 2, 2, 2],
                           [3, 3, 3, 3],
                           [4, 4, 4, 4]])

        actual = tr.zero_mean(X_test, X_train)

        expected = np.array([[-2.5,-3.5,-4.,-3.25],
                             [-1.5,-2.5,-3.,-2.25],
                             [-0.5,-1.5,-2.,-1.25],
                             [ 0.5,-0.5,-1.,-0.25]])
        
        npt.assert_equal(actual, expected)

    def test_compute_covariance(self):
        X = np.array([[2, 4, 6],
                      [3, 4, 5],
                      [4, 5, 4],
                      [5, 6, 3],
                      [6, 6, 2]])

        actual_C = tr.compute_cov(X.T)
        expected_C = np.array([[2.5, 1.5, -2.5],
                               [1.5, 1.0, -1.5],
                               [-2.5, -1.5, 2.5]])

        npt.assert_equal(actual_C, expected_C)

    def test_covariance_is_symmetric(self):
        X = np.arange(200).reshape(20, 10)
        C = tr.compute_cov(X)

        npt.assert_array_equal(C, C.T)

    def test_compute_eigenpairs(self):
        C = np.array([[0.0, 0.5, 1.0],
                      [0.5, 0.5, 0.5],
                      [1.0, 0.5, 0.0]])

        (actual_val, actual_vec) = tr.compute_eigenvalues(C)

        expected_val = np.array([-1.0, 0.0, 1.5])
        expected_vec = np.array([[ sqrt(1./2.),  sqrt(1./6.), -sqrt(1./3.)],
                                 [ 0.000000000, -sqrt(2./3.), -sqrt(1./3.)],
                                 [-sqrt(1./2.),  sqrt(1./6.), -sqrt(1./3.)]])

        npt.assert_allclose(actual_val, expected_val, atol=1e-7)
        npt.assert_allclose(actual_vec, expected_vec, atol=1e-7)

    def test_sort_eigenpairs(self):
        eigenvalues = np.array([1, 3, 4, 2])
        eigenvectors = np.array([[1, 3, 4, 2],
                                 [1, 3, 4, 2],
                                 [1, 3, 4, 2],
                                 [1, 3, 4, 2]])
                                 
        expected_sorted_eigenvalues = np.array([4, 3, 2, 1])
        expected_sorted_eigenvectors = np.array([[4, 3, 2, 1],
                                                 [4, 3, 2, 1],
                                                 [4, 3, 2, 1],
                                                 [4, 3, 2, 1]])
                                                 
        actual_sorted_eigenvalues, actual_sorted_eigenvectors = tr.sort_eigenpairs(eigenvalues, eigenvectors)
        
        npt.assert_equal(actual_sorted_eigenvalues, expected_sorted_eigenvalues)                                         
        npt.assert_equal(actual_sorted_eigenvectors, expected_sorted_eigenvectors)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_divide_dataset_into_train_and_test_sets']
    unittest.main()

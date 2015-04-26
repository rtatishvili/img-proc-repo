import unittest
import src.histogram as hist
import numpy as np

class Test(unittest.TestCase):


    def test_return_zero_array_for_empty_image(self):

        image_array = []
        actual = hist.calc_prob_density(image_array)
        expected = np.zeros(len(actual))
        
        np.testing.assert_equal(actual, expected, 'Probability density of an empty image should contain only zeroes')


    def test_by_default_return_array_of_length_256(self):
        
        image_array = [0, 1, 2]
        result = hist.calc_prob_density(image_array)
        actual = len(result)
        expected = 256

        np.testing.assert_equal(actual, expected, 'Probability density should have 256 elements')
        
    
    def test_return_array_of_length_of_range(self):
        
        image_array = [0, 0, 1, 1, 2, 2, 3, 3]
        result = hist.calc_prob_density(image_array, 2, 4)
        actual = len(result)
        expected = 2

        np.testing.assert_equal(actual, expected, 'Probability density with range 2 to 4 should have 2 elements')

        
    def test_return_array_with_sum_of_one(self):
        
        image_array = [0, 0, 1, 1, 2, 2, 3, 3]
        result = hist.calc_prob_density(image_array)
        
        actual = np.sum(result)
        expected = 1.0
        
        np.testing.assert_equal(actual, expected, 'Probability density should sum up to one')


    def test_return_array_with_sum_of_one_for_range(self):
        
        image_array = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3]
        result = hist.calc_prob_density(image_array, 2, 4)
        actual = np.sum(result)
        expected = 1.0
        
        np.testing.assert_equal(actual, expected, 'Probability density for range should sum up to one')
        

    def test_return_probability_for_range(self):
        
        image_array = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3]
        actual = hist.calc_prob_density(image_array, 2, 4)
        expected = [0.25, 0.75]
        
        np.testing.assert_equal(actual, expected, 'Probability density should compute probabilities within the range')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
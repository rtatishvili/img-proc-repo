import unittest
import src.quantisation as q
import src.histogram as hist
import numpy as np

class Test(unittest.TestCase):


    def test_compute_weighted_square_error(self):
        
        image_array = [0, 0, 64, 64, 128, 128, 192, 192, 192, 192, 192, 192]
        histogram = hist.calc_histogram(image_array)
        prob_density = hist.calc_prob_density(image_array)
        intervals = [0, 129, 256]
        points = [64, 191]
        actual = q.square_error(intervals, points, prob_density, histogram)
        expected = float(64**2 * 4) * (1.0 / 6.0) + float(1**2 * 6) * (1.0 / 2.0)
        
        np.testing.assert_equal(actual, expected, 'Error measure is not correct')
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_compute_weighted_square_error']
    unittest.main()
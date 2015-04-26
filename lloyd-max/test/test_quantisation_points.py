import unittest
import numpy as np
import src.quantisation as q

class Test(unittest.TestCase):

    
    def test_init_by_default_count_to_8(self):
        
        points = q.init_quantisation_points()
        actual = len(points)
        expected = 8
        
        np.testing.assert_equal(actual, expected, 'Quantisation points should create 8 intervals by default')

    def test_init_produce_interval_middle_points(self):
        
        actual = q.init_quantisation_points(3)        
        expected = [0 * 256 / 3 + 256 / 6,
                    1 * 256 / 3 + 256 / 6,
                    2 * 256 / 3 + 256 / 6]
        
        np.testing.assert_array_equal(expected, actual, 'Quantisation points are not correct')
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_produce_interval_middle_points']
    unittest.main()
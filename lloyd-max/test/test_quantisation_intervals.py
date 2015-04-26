import unittest
import numpy as np
import src.quantisation as q

class Test(unittest.TestCase):


    def test_init_by_default_count_to_8(self):
        
        intervals = q.init_quantisation_intervals()
        expected = 9 # 9 boundaries means 8 intervals
        actual = len(intervals)
        
        np.testing.assert_equal(expected, actual, 'Quantisation intervals should create 8 intervals by default')

    
    def test_init_have_boundaries_at_almost_same_distance(self):

        actual = q.init_quantisation_intervals(3)
        expected = [0, 256 / 3, 256 * 2 / 3, 256]
        
        np.testing.assert_array_equal(expected, actual, 'Quantisation intervals do not have boundaries at almost same distance')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_intervals']
    unittest.main()
import unittest
import numpy as np
from calc.distance import euclidean_2d_array

class Test(unittest.TestCase):


    def test_euclidean_2d_array(self):
        test_array = np.ones((5, 5))
        
        result_array = euclidean_2d_array(test_array.shape, (0, 0))
        
        expected = np.array([[0, 1, 2, 3, 4],
                            [1, np.sqrt(2), np.sqrt(5), np.sqrt(10), np.sqrt(17)],
                            [2, np.sqrt(5), np.sqrt(8), np.sqrt(13), np.sqrt(20)],
                            [3, np.sqrt(10), np.sqrt(13), np.sqrt(18), 5],
                            [4, np.sqrt(17), np.sqrt(20), 5, np.sqrt(32)]])

        np.testing.assert_array_equal(result_array, expected, "Euclidean distance not correct")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_euclidean_2d_array']
    unittest.main()
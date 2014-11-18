import unittest
import numpy as np
from image_op.ring_mask import RingMask 

def func(size):
    condition_list = []
    for i in range(size[0]):
        condition_sublist = []
        for j in range(size[1]):
            condition_sublist.append(i == j)
        condition_list.append(condition_sublist)

    return np.array(condition_list)

class Test(unittest.TestCase):

    
    def test_ring_mask_radius_coverage(self):
        # --------- Test with euclidean
        test_array = np.ones((5, 5))

        ring_mask = RingMask(test_array.shape, lambda x: x * 0, lambda x: x, 1.0, 2.1, (2, 2))
      
        expected_array = np.array([[ 1, 1, 0, 1, 1],
                                   [ 1, 0, 0, 0, 1],
                                   [ 0, 0, 1, 0, 0],
                                   [ 1, 0, 0, 0, 1],
                                   [ 1, 1, 0, 1, 1]])    
        
        masked_array = ring_mask.apply_mask(test_array)
        
        np.testing.assert_array_equal(masked_array, expected_array, "Mask did not apply function")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_mask_initialization']
    unittest.main()
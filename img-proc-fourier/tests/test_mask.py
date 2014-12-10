import unittest

import numpy as np

from common.image_op.mask import Mask


class Test(unittest.TestCase):
    
    def test_mask_with_multiplier_function(self):
        # --------- Test with euclidean
        test_array = np.ones((5,5))
        
        mask = Mask(test_array.shape, lambda x: x * 2, lambda x: x)
    
        condition = np.ones((5, 5), dtype=np.bool)
    
        mask.set_data(condition)
        
        masked_array = mask.apply_mask(test_array)
        
        np.testing.assert_array_equal(masked_array, np.ones((5, 5)) * 2, "Mask did not apply function")

    
    def test_mask_with_2d_condition(self):
        # --------- Test with euclidean
        test_array = np.ones((5, 5))

        mask = Mask(test_array.shape, lambda x: x * 1, lambda x: x * 0)
     
        condition = np.array([[ True, True, True, True, True],
                             [ True, True, False, True, True],
                             [ True, False, False, False, True],
                             [ True, True, False, True, True],
                             [ True, True, True, True, True]])    
        
        expected_array = np.array([[ 1, 1, 1, 1, 1],
                                   [ 1, 1, 0, 1, 1],
                                   [ 1, 0, 0, 0, 1],
                                   [ 1, 1, 0, 1, 1],
                                   [ 1, 1, 1, 1, 1]])    
       
        mask.set_data(condition)
        
        masked_array = mask.apply_mask(test_array)
        
        np.testing.assert_array_equal(masked_array, expected_array, "Mask did not apply function")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_mask_initialization']
    unittest.main()
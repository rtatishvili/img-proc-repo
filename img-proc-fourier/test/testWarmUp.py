'''
Created on Oct 29, 2014

@author: revaz
'''
import unittest
from fourier.Helper import Helper

class Test(unittest.TestCase):


    def test_euclidean_distance(self):
        
        self.assertAlmostEqual(Helper.euclidean_distance([1, 1], [2, 2]), 1.414, 3)
        self.assertAlmostEqual(Helper.euclidean_distance([1,1,1], [2,2,2]), 1.732, 3)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
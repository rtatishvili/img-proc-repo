'''
Created on Oct 29, 2014

@author: revaz
'''
import unittest
from fourier.Helper import Helper

class Test(unittest.TestCase):


    def testDistanceGivenNumbersXYShouldReturnEuclideanDistance(self):
        x1 = 1
        y1 = 1
        x2 = 2
        y2 = 2
        
        result = Helper.distance(px = x1, py = y1, qx = x2, qy = y2)
        print result
        self.assertAlmostEqual(result, 1.414, 3)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
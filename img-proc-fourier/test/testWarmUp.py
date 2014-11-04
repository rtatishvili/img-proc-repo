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
        self.assertRaises(RuntimeError, Helper.euclidean_distance, [1,1,1], [22,2])
        
        
    def test_draw_circle(self):
        im = Helper.read_image('bauckhage.jpg', as_array=True)
        newImageArray = Helper.draw_circle(im, 30, 50, True) 
        newImage = Helper.create_image(newImageArray)
        newImage.show()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
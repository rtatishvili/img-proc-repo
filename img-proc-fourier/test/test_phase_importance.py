'''
Created on Nov 10, 2014

@author: kostadin
'''
import unittest
import math as math
import numpy as np

import fourier.image_io as image_io


class Test(unittest.TestCase):


    def test_phase_importance(self):
        pass
        im = image_io.read_image('bauckhage.jpg', as_array=False)
        ft = np.fft.fft2(im)
        print ft[0, 0]
        print np.real(ft[0, 0]) 
        print np.imag(ft[0, 0])
        print math.sqrt(np.real(ft[0, 0]) ** 2 + np.imag(ft[0, 0]) ** 2)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPhase']
    unittest.main()
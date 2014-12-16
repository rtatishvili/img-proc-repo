__author__ = 'myrmidon'

import unittest
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift

class MyTestCase(unittest.TestCase):
    image = np.zeros((5, 5))
    image[x for x in image.shape] = 1


if __name__ == '__main__':
    unittest.main()

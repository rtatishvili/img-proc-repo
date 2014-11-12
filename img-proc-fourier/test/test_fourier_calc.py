'''
Created on Nov 10, 2014

@author: kostadin
'''
import unittest
import fourier.fourier_calc as fourier_calc
import fourier.image_io as image_io
import numpy as np
from cmath import phase


class Test(unittest.TestCase):


    def test_magnitude(self):
        error_msg = 'Wrong magnitude extraction'
        self.assertEqual(fourier_calc.magnitude(complex(10, 12)), 15.620499351813308, error_msg)
        self.assertEqual(fourier_calc.magnitude(complex(10, 0)), 10, error_msg)
        self.assertEqual(fourier_calc.magnitude(complex(0, 12)), 12, error_msg)
        
    def test_phase(self):
        error_msg = 'Wrong phase extraction'
        self.assertEqual(fourier_calc.phase(complex(10, 12)), 0.8760580505981934, error_msg)
        self.assertEqual(fourier_calc.phase(complex(10, 0)), 0, error_msg)
        self.assertEqual(fourier_calc.phase(complex(0, 12)), 1.5707963267948966, error_msg)
        
    def test_create_complex_number(self):
        error_msg = 'Wrong complex number'
        im = image_io.read_image('bauckhage.jpg', as_array=False)
        ft = np.fft.fft2(im)
        for i in range(ft.shape[0]):
            for j in range(ft.shape[1]):
                magnitude = fourier_calc.magnitude(ft[i, j])
                phase = fourier_calc.phase(ft[i, j])
                self.assertAlmostEqual(ft[i, j], fourier_calc.create_complex_number(magnitude, phase), 8, error_msg)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_fourier_calc']
    unittest.main()
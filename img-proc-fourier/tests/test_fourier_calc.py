import unittest

import numpy as np

from calc import fourier_calc as fourier_calc


class Test(unittest.TestCase):


    def test_magnitude(self):
        error_msg = 'Wrong magnitude extraction'
        
        c1 = np.array([complex(10, 12)])
        c2 = np.array([complex(10, 0)])
        c3 = np.array([complex(0, 12)])
        
        self.assertEqual(fourier_calc.magnitude(c1)[0], 15.620499351813308, error_msg)
        self.assertEqual(fourier_calc.magnitude(c2)[0], 10, error_msg)
        self.assertEqual(fourier_calc.magnitude(c3)[0], 12, error_msg)
        
    def test_phase(self):
        error_msg = 'Wrong phase extraction'
        
        c1 = np.array([complex(10, 12)])
        c2 = np.array([complex(10, 0)])
        c3 = np.array([complex(0, 12)])
        
        self.assertEqual(fourier_calc.phase(c1)[0], 0.8760580505981934, error_msg)
        self.assertEqual(fourier_calc.phase(c2)[0], 0, error_msg)
        self.assertEqual(fourier_calc.phase(c3)[0], 1.5707963267948966, error_msg)
        
    #def test_create_complex_array(self):
    #    error_msg = 'Wrong complex number'
    #    im = image_io.read_image('../Resources/bauckhage.jpg', as_array=False)
    #    ft = np.fft.fft2(im)
        
    #    magnitude = fourier_calc.magnitude(ft)        
    #    phase = fourier_calc.phase(ft)        
    #    result = np.allclose(ft, fourier_calc.create_complex_array(magnitude, phase))
        
    #    self.assertTrue(result, error_msg)
                
                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_fourier_calc']
    unittest.main()
    
    
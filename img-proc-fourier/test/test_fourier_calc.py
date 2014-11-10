'''
Created on Nov 10, 2014

@author: kostadin
'''
import unittest
import fourier.fourier_calc as fourier_calc


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
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_fourier_calc']
    unittest.main()
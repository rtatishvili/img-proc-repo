import unittest

import image_op.image_io as io
import calc.bicubic_interpolation as bci
import math


class MyTestCase(unittest.TestCase):
    def test_something(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        print image[1:5, 0:4]
        result = bci.resample(image, 1.8, 2.3)
        assert result == 109.670134

if __name__ == '__main__':
    unittest.main()

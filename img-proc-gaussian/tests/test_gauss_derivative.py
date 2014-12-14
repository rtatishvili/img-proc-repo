__author__ = 'kostadin'

import unittest

import numpy as np
import calc.gaussian_mask as gm
import image_op.image_io as io
import image_op.image_manip as im


class MyTestCase(unittest.TestCase):
    def test_gauss_derivative(self):
        image = io.read_image("../Resources/asterixGrey.jpg", as_array=True)
        img_dx, img_dy = gm.gauss_derivatives(image, (9, 9))
          # np.testing.assert_array_almost_equal(img_dx, img_dy, 5)
        io.save_array_as_gray_image(img_dx, "../Generated/asterixGrey_dx.jpg")
        io.save_array_as_gray_image(img_dx, "../Generated/asterixGrey_dy.jpg")
        io.save_array_as_gray_image(np.sqrt(np.power(img_dx, 2) + np.power(img_dy, 2)), "../Generated/asterixGrey_magnitude.jpg")
if __name__ == '__main__':
    unittest.main()

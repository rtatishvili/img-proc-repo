__author__ = 'kostadin'

import unittest

import numpy as np
import calc.gaussian_mask as gm
import image_op.image_io as io
import image_op.image_manip as im


class MyTestCase(unittest.TestCase):
    def test_gauss_derivative(self):
        image = io.read_image("../Resources/clock.jpg", as_array=True)
        img_dx, img_dy = gm.gauss_derivatives(image, (9, 9))

        image_smooth = im.apply_matrix_mask(image, gm.generate_gauss_2d(size=(9, 9)))
        image_diff = image_smooth - image

        io.save_array_as_gray_image(img_dx, "../Generated/clock_dx.jpg", normalize=True)
        io.save_array_as_gray_image(img_dy, "../Generated/clock_dy.jpg", normalize=True)
        io.save_array_as_gray_image(np.sqrt(np.power(img_dx, 2) + np.power(img_dy, 2)),
                                    "../Generated/clock_magnitude.jpg", normalize=True)
        io.save_array_as_gray_image(np.abs(image_diff), "../Generated/clock_diff.jpg", normalize=True )

if __name__ == '__main__':
    unittest.main()

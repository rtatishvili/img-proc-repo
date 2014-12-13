__author__ = 'kostadin'

import unittest

import calc.gaussian_mask as gm
import image_op.image_io as io
import image_op.image_manip as im
import time

class MyTestCase(unittest.TestCase):

    def test_gaussian_blur_vector(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        array_x = gm.generate_gauss_1D(size=15)
        array_y = gm.generate_gauss_1D(size=15)
        t = time.time()
        new_image = im.applay_array_mask(image, array_x, array_y)
        print time.time() - t
        print new_image[0, 0]
        io.save_array_as_gray_image(new_image, "../Generated/bauckhage11.jpg")

    def test_gaussian_blur_matrix(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)

        mask = gm.generate_gauss_2D((15, 15))
        t = time.time()
        new_image = im.applay_matrix_mask(image, mask)
        print time.time() - t
        print new_image[0, 0]
        io.save_array_as_gray_image(new_image, "../Generated/bauckhage1.jpg")
        io.save_array_as_gray_image(new_image - image, "../Generated/bauckhage2.jpg")


if __name__ == '__main__':
    unittest.main()

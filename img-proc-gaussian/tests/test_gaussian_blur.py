import numpy

__author__ = 'kostadin'

import unittest
import numpy as np

import calc.gaussian_mask as gm
import image_op.image_io as io
import image_op.image_manip as im
import time


class MyTestCase(unittest.TestCase):
    def test_gauss_mask(self):
        mask_2d = gm.generate_gauss_2d((3, 3))
        mask_1d_x = gm.generate_gauss_1d(3)
        mask_1d_y = gm.generate_gauss_1d(3, True)
        np.testing.assert_array_equal(mask_2d, mask_1d_x*mask_1d_y)

    def test_apply(self):
        window = np.arange(49).reshape((7, 7))
        mask = gm.generate_gauss_1d(3)
        print window
        # print mask
        # print sum(sum(window * mask))
        print im.extract_window(window, (1, 3), (2, 4))


    def test_gaussian_blur_vector(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        array_x = gm.generate_gauss_1d(size=15)
        array_y = gm.generate_gauss_1d(size=15)
        t = time.time()
        new_image = im.apply_array_mask(image, array_x, array_y)
        print time.time() - t
        print "test1:" + str(new_image[0, 0])
        io.save_array_as_gray_image(new_image, "../Generated/bauckhage11.jpg")
    #
    def test_gaussian_blur_matrix(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)

        mask = gm.generate_gauss_2d((15, 15))
        t = time.time()
        new_image = im.apply_matrix_mask(image, mask)
        print time.time() - t
        print "test2:" + str(new_image[0, 0])
        io.save_array_as_gray_image(new_image, "../Generated/bauckhage1.jpg")
        io.save_array_as_gray_image(new_image - image, "../Generated/bauckhage2.jpg")

    def test_as(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        array_x = gm.generate_gauss_1d(size=15)
        array_y = gm.generate_gauss_1d(size=15)

        mask = gm.generate_gauss_2d((15, 15))
        t = time.time()
        new_image_2d = im.apply_matrix_mask(image, mask)
        print "2D: " + str(time.time() - t)
        t1 = time.time()
        new_image_1d = im.apply_array_mask(image, array_x, array_y)
        print "1D: " + str(time.time() - t1)

        io.save_array_as_gray_image(new_image_2d, "../Generated/bauckhage1.jpg")
        io.save_array_as_gray_image(new_image_2d - image, "../Generated/bauckhage2.jpg")
        io.save_array_as_gray_image(new_image_1d, "../Generated/bauckhage11.jpg")

        np.testing.assert_array_almost_equal(new_image_1d, new_image_2d, 6)

if __name__ == '__main__':
    unittest.main()

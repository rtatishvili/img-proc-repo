
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

    def test_frequency_domain(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        mask_2d = gm.generate_gauss_2d((15, 15))
        mask = gm.zero_pad_mask(mask_2d, image.shape)
        image_fft = np.fft.fftshift(np.fft.fft2(image))
        mask_fft = np.fft.fftshift(np.fft.fft2(mask))

        new_image_fft = image_fft * mask_fft
        new_image = np.fft.ifftshift(np.fft.ifft2(new_image_fft))
        io.save_array_as_gray_image(np.abs(new_image), "../Generated/bauckhage_frequnecy.jpg")
        np.testing.assert_equal(gm.zero_pad_mask(mask_2d, (256, 256)).shape, (256, 256))

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

        t2 = time.time()
        mask_zero = gm.zero_pad_mask(mask, image.shape)
        new_image_freq = im.apply_fourier_mask(image, mask_zero)
        print "freq: " + str(time.time() - t2)


        io.save_array_as_gray_image(np.abs(new_image_freq), "../Generated/bauckhage_frequnecy.jpg")
        io.save_array_as_gray_image(new_image_2d, "../Generated/bauckhage_2s.jpg")
        io.save_array_as_gray_image(new_image_2d - image, "../Generated/bauckhage_edge.jpg")
        io.save_array_as_gray_image(new_image_1d, "../Generated/bauckhage_1d.jpg")

        np.testing.assert_array_almost_equal(new_image_1d, new_image_2d, 6)



        np.testing.assert_array_almost_equal(new_image_1d[10:-10, 10:-10], np.abs(new_image_freq[10:-10, 10:-10]), 6)
#         np.testing.assert_array_almost_equal(new_image_1d, np.abs(new_image_freq), 6)

if __name__ == '__main__':
    unittest.main()

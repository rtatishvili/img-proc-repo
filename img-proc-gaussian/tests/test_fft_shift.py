__author__ = 'myrmidon'

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftn, ifftn


class MyTestCase(unittest.TestCase):
    image = np.arange(49).reshape(7, 7)

    # assert_array_equal
    def test_fft2_ifft2_equal(self):
        assert_array_equal(ifft2(fft2(self.image)), self.image)

    def test_fftn_ifftn_equal(self):
        assert_array_equal(ifftn(fftn(self.image)), self.image)

    def test_fft2_fftshift_ifftshift_ifft2_equal(self):
        assert_array_equal(ifft2(ifftshift(fftshift(fft2(self.image)))), self.image)

    def test_fft2_fftshift_ifft2_ifftshift_equal(self):
        assert_array_equal(ifftshift(ifft2(fftshift(fft2(self.image)))), self.image)

    def test_fftn_fftshift_ifftshift_ifftn_equal(self):
        assert_array_equal(ifftn(ifftshift(fftshift(fftn(self.image)))), self.image)

    def test_fftn_fftshift_ifftn_ifftshift_equal(self):
        assert_array_equal(ifftshift(ifftn(fftshift(fftn(self.image)))), self.image)

    # assert array equal absolute
    def test_fft2_ifft2_equal_abs(self):
        assert_array_equal(abs(ifft2(fft2(self.image))), self.image)

    def test_fftn_ifftn_equal_abs(self):
        assert_array_equal(abs(ifftn(fftn(self.image))), self.image)

    def test_fft2_fftshift_ifftshift_ifft2_equal_abs(self):
        assert_array_equal(abs(ifft2(ifftshift(fftshift(fft2(self.image))))), self.image)

    def test_fft2_fftshift_ifft2_ifftshift_equal_abs(self):
        assert_array_equal(abs(ifftshift(ifft2(fftshift(fft2(self.image))))), self.image)

    def test_fftn_fftshift_ifftshift_ifftn_equal_abs(self):
        assert_array_equal(abs(ifftn(ifftshift(fftshift(fftn(self.image))))), self.image)

    def test_fftn_fftshift_ifftn_ifftshift_equal_abs(self):
        assert_array_equal(abs(ifftshift(ifftn(fftshift(fftn(self.image))))), self.image)

    # assert array almost equal
    def test_fft2_ifft2_equal_almost(self):
        assert_array_almost_equal(ifft2(fft2(self.image)), self.image)

    def test_fftn_ifftn_equal_almost(self):
        assert_array_almost_equal(ifftn(fftn(self.image)), self.image)

    def test_fft2_fftshift_ifftshift_ifft2_equal_almost(self):
        assert_array_almost_equal(ifft2(ifftshift(fftshift(fft2(self.image)))), self.image)

    def test_fft2_fftshift_ifft2_ifftshift_equal_almost(self):
        assert_array_almost_equal(ifftshift(ifft2(fftshift(fft2(self.image)))), self.image)

    def test_fftn_fftshift_ifftshift_ifftn_equal_almost(self):
        assert_array_almost_equal(ifftn(ifftshift(fftshift(fftn(self.image)))), self.image)

    def test_fftn_fftshift_ifftn_ifftshift_equal_almost(self):
        assert_array_almost_equal(ifftshift(ifftn(fftshift(fftn(self.image)))), self.image)

    # assert array equal almost abs
    def test_fft2_ifft2_equal_almost_abs(self):
        assert_array_almost_equal(abs(ifft2(fft2(self.image))), self.image)

    def test_fftn_ifftn_equal_almost_abs(self):
        assert_array_almost_equal(abs(ifftn(fftn(self.image))), self.image)

    def test_fft2_fftshift_ifftshift_ifft2_equal_almost_abs(self):
        assert_array_almost_equal(abs(ifft2(ifftshift(fftshift(fft2(self.image))))), self.image)

    def test_fft2_fftshift_ifft2_ifftshift_equal_almost_abs(self):
        assert_array_almost_equal(abs(ifftshift(ifft2(fftshift(fft2(self.image))))), self.image)

    def test_fftn_fftshift_ifftshift_ifftn_equal_almost_abs(self):
        assert_array_almost_equal(abs(ifftn(ifftshift(fftshift(fftn(self.image))))), self.image)

    def test_fftn_fftshift_ifftn_ifftshift_equal_almost_abs(self):
        assert_array_almost_equal(abs(ifftshift(ifftn(fftshift(fftn(self.image))))), self.image)

    # others
    def test_fft_ifft_different_shape(self):
        ft_image = fft2(self.image, (256, 256))
        result = ifft2(ft_image, self.image.shape)

        assert_array_almost_equal(abs(result), self.image)

if __name__ == '__main__':
    unittest.main()

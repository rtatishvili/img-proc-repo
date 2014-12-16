__author__ = 'myrmidon'

import unittest
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import fftconvolve, convolve2d
from numpy.testing import assert_array_equal, assert_array_almost_equal
from calc.gaussian_mask import zero_pad_mask


class TestMultiplyFFT(unittest.TestCase):
    image = np.arange(5 ** 2).reshape(5, 5)

    # Creating a mask that does nothing on convolution
    inert_mask = np.zeros((3, 3))
    inert_mask[inert_mask.shape[0] // 2, inert_mask.shape[0] // 2] = 1

    @staticmethod
    def pad_with_zeros(array, desired_shape):
        current_shape = np.asarray(array.shape)
        final_shape = np.asarray(desired_shape)

        ranges = (tuple(final_shape // 2 - current_shape // 2), tuple(final_shape // 2 + current_shape // 2 + 1))

        x, y = np.ogrid[ranges[0][0]:ranges[1][0], ranges[0][1]:ranges[1][1]]

        new_array = np.zeros(desired_shape)
        new_array[x, y] = array

        return new_array

    # Creating padded inert mask
    inert_mask_padded = np.zeros(image.shape)
    x, y = np.ogrid[inert_mask_padded.shape[0] // 2 - inert_mask.shape[0] // 2: inert_mask_padded.shape[0] // 2 +
                                                                                inert_mask.shape[0] // 2 + 1,
           inert_mask_padded.shape[1] // 2 - inert_mask.shape[1] // 2: inert_mask_padded.shape[1] // 2 +
                                                                       inert_mask.shape[1] // 2 + 1]
    inert_mask_padded[x, y] = inert_mask

    # kosta's padded mask
    inert_mask_padded_kosta = zero_pad_mask(inert_mask, image.shape)

    def test_inert_mask_conv2d(self):
        result = convolve2d(self.image, self.inert_mask, mode='same')

        assert_array_equal(result, self.image)

    def test_inert_mask_fftconv_same(self):
        result = fftconvolve(self.image, self.inert_mask, mode='same')

        assert_array_almost_equal(result, self.image)

    def test_our_comp_real(self):
        # Now, our computation
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask, self.image.shape)
        ft_res_ft = ft_image * ft_mask

        result = ifft2(ft_res_ft)

        assert_array_equal(result.real, self.image)

    def test_our_comp_abs(self):
        # Now, our computation
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask, self.image.shape)
        ft_res_ft = ft_image * ft_mask

        result = ifft2(ft_res_ft)

        print result
        print self.image
        print abs(result)

        assert_array_equal(abs(result), self.image)

    def test_our_comp_shifted_abs(self):
        # Now, our computation
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask, self.image.shape)
        ft_res_ft = fftshift(ft_image) * fftshift(ft_mask)

        result = ifft2(ifftshift(ft_res_ft))

        assert_array_equal(abs(result), self.image)

    def test_kost_comp(self):
        ft_image = fft2(self.image)
        ft_mask_padded = fft2(self.inert_mask_padded)

        ft_result = ft_image * ft_mask_padded

        result = ifft2(ft_result)

        assert_array_equal(result, self.image)

    def test_kost_comp_real(self):
        ft_image = fft2(self.image)
        ft_mask_padded = fft2(self.inert_mask_padded)

        ft_result = ft_image * ft_mask_padded

        result = ifft2(ft_result)

        assert_array_equal(result.real, self.image)

    def test_kost_comp_abs(self):
        ft_image = fft2(self.image)
        ft_mask_padded = fft2(self.inert_mask_padded)

        ft_result = fftshift(ft_image) * fftshift(ft_mask_padded)

        result = ifftshift(ifft2(ft_result))

        assert_array_almost_equal(abs(result), self.image)

    def test_padded_conv2d(self):
        result = convolve2d(self.image, self.inert_mask_padded, mode='same')

        assert_array_equal(result, self.image)

    def test_padded_fftconv(self):
        result = fftconvolve(self.image, self.inert_mask_padded, mode='same')
        # result[result < 1] = 0

        assert_array_almost_equal(result, self.image)

    def test_kosta_comp(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        ft_result = ft_image * ft_mask
        result = ifft2(ft_result)

        assert_array_almost_equal(result, self.image)

    def test_kosta_comp_shifted(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        sh_ft_image = fftshift(ft_image)
        sh_ft_mask = fftshift(ft_mask)

        sh_ft_res = sh_ft_image * sh_ft_mask

        result = ifft2(ifftshift(sh_ft_res))

        assert_array_almost_equal(result, self.image)

    def test_kosta_comp_shifted_wrong(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        sh_ft_image = fftshift(ft_image)
        sh_ft_mask = fftshift(ft_mask)

        sh_ft_res = sh_ft_image * sh_ft_mask

        result = ifftshift(ifft2(sh_ft_res))

        assert_array_almost_equal(result, self.image)

    def test_kosta_comp_real(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        ft_result = ft_image * ft_mask
        result = ifft2(ft_result)

        assert_array_almost_equal(result.real, self.image)

    def test_kosta_comp_shifted_real(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        sh_ft_image = fftshift(ft_image)
        sh_ft_mask = fftshift(ft_mask)

        sh_ft_res = sh_ft_image * sh_ft_mask

        result = ifft2(ifftshift(sh_ft_res))

        assert_array_almost_equal(result.real, self.image)

    def test_kosta_comp_shifted_wrong_real(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        sh_ft_image = fftshift(ft_image)
        sh_ft_mask = fftshift(ft_mask)

        sh_ft_res = sh_ft_image * sh_ft_mask

        result = ifftshift(ifft2(sh_ft_res))

        assert_array_almost_equal(result.real, self.image)

    def test_kosta_comp_abs(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        ft_result = ft_image * ft_mask
        result = ifft2(ft_result)

        assert_array_almost_equal(abs(result), self.image)

    def test_kosta_comp_shifted_abs(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        sh_ft_image = fftshift(ft_image)
        sh_ft_mask = fftshift(ft_mask)

        sh_ft_res = sh_ft_image * sh_ft_mask

        result = ifft2(ifftshift(sh_ft_res))

        assert_array_almost_equal(abs(result), self.image)

    def test_kosta_comp_shifted_wrong_abs(self):
        ft_image = fft2(self.image)
        ft_mask = fft2(self.inert_mask_padded_kosta)

        sh_ft_image = fftshift(ft_image)
        sh_ft_mask = fftshift(ft_mask)

        sh_ft_res = sh_ft_image * sh_ft_mask

        result = ifftshift(ifft2(sh_ft_res))

        assert_array_almost_equal(abs(result), self.image)

    def test_mask_padding(self):
        assert_array_equal(self.inert_mask_padded, self.inert_mask_padded_kosta)

    def test_padding(self):
        assert_array_equal(self.pad_with_zeros(self.inert_mask, self.image.shape), self.inert_mask_padded)

    def test_why_does_this_not_work(self):
        ft_image = fft2(self.image)

if __name__ == '__main__':
    unittest.main()

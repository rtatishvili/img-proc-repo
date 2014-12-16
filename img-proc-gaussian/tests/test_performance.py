__author__ = 'kostadin'

import unittest

import calc.gaussian_mask as gm
import image_op.image_io as io
import image_op.image_manip as im
import matplotlib.pyplot as plt

import time
import numpy as np


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = 10
        cls.image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        cls.masks_2d = [gm.generate_gauss_2d((size, size)) for size in range(3, 26) if size % 2 == 1]
        cls.masks_1d = [gm.generate_gauss_1d(size) for size in range(3, 26) if size % 2 == 1]
        cls.time_results = np.zeros(4*len(cls.masks_2d)).reshape((len(cls.masks_2d), 4))

    @classmethod
    def tearDownClass(cls):
        print cls.time_results
        labels = ['Fourier filter', '2D - Convolution', '1D - Convolution']
        plt.title('Performance of different implementation of Gaussian Filter').set_fontsize(26)
        plt.xlabel('Mask dimension', fontsize=18)
        plt.ylabel('time (s)', fontsize=18)
        # plt.yscale('log')
        plt.xticks(cls.time_results.T[0])
        plt.plot(cls.time_results.T[0], cls.time_results.T[3])
        plt.plot(cls.time_results.T[0], cls.time_results.T[2])
        plt.plot(cls.time_results.T[0], cls.time_results.T[1])
        plt.legend(labels, loc='upper left',
                   labelspacing=0.0, handletextpad=0.0,
                   handlelength=1.5,
                   fancybox=True, shadow=True, fontsize=14)
        plt.show()

    def test_1d_convolution(self):
        print "----- 1d convolution -----"
        count = 0
        for mask in self.masks_1d:
            time_sum = 0.
            for i in range(self.times):
                temp_time = time.clock()
                im.apply_array_mask(self.image, mask, mask)
                time_sum += time.clock() - temp_time
            self.time_results[count, 0] = mask.shape[0]
            self.time_results[count, 1] = time_sum / float(self.times)
            count += 1
            print 'Mask size ' + str(mask.shape[0]) + ', in time:' + str(time_sum / float(self.times))

    def test_2d_convolution(self):
        print "----- 2d convolution -----"
        count = 0
        for mask in self.masks_2d:
            time_sum = 0.
            for i in range(self.times):
                temp_time = time.clock()
                im.apply_matrix_mask(self.image, mask)
                time_sum += time.clock() - temp_time
            self.time_results[count, 0] = mask.shape[0]
            self.time_results[count, 2] = time_sum / float(self.times)
            count += 1
            print 'Mask size ' + str(mask.shape[0]) + ', in time:' + str(time_sum / float(self.times))

    def test_fourier_filtering(self):
        print "----- fourier filtering -----"
        count = 0
        for mask in self.masks_2d:
            mask_padded = gm.zero_pad_mask(mask, self.image.shape)
            time_sum = 0.
            for i in range(self.times):
                temp_time = time.clock()
                im.apply_fourier_mask(self.image, mask_padded)
                time_sum += time.clock() - temp_time
            self.time_results[count, 0] = mask.shape[0]
            self.time_results[count, 3] = time_sum / float(self.times)
            count += 1
            print 'Mask size ' + str(mask.shape[0]) + ', in time:' + str(time_sum / float(self.times))

if __name__ == '__main__':
    unittest.main()
'''
Created on Nov 4, 2014

@author: kostadin
'''
import unittest
import numpy as np

from fourier.Helper import Helper
#import matplotlib.pyplot as plt

class Test(unittest.TestCase):


    def test_band_pass_filter(self):
        pass
        im = Helper.read_image('bauckhage.jpg', as_array=False)
        ft = np.fft.fft2(im)
        sh = np.fft.fftshift(ft)
        Helper.save_image(np.log(np.abs(sh)), 'FourierTransformation.jpg')
        
#         plt.title('Fourier Transformation'), plt.xticks([]), plt.yticks([])
#         plt.imshow(np.log(np.abs(sh)), cmap = plt.get_cmap('gray'))
#         plt.show()
       
        picture = Helper.draw_circle(sh, 20, 50, False)
        Helper.save_image(np.log(np.abs(picture)), 'FrequencySuppression.jpg')
        
#         plt.title('Frequency Suppression'), plt.xticks([]), plt.yticks([])
#         plt.imshow(np.log(np.abs(picture)), cmap = plt.get_cmap('gray'))
#         plt.show()
       
        
        inverse_p = np.fft.ifft2(picture)
        Helper.save_image(np.abs(inverse_p), 'InverseFourierTransformation.jpg')
#         plt.title('Inverse Fourier Transformation'), plt.xticks([]), plt.yticks([])
#         plt.imshow(np.abs(inverse_p), cmap = plt.get_cmap('gray'))
#         plt.show()
    
    def test_fourier_transformation(self):
        pass
#         im = Helper.read_image('bauckhage.jpg', as_array=False)
#         ft = np.fft.fft2(im)
#         ift = np.fft.fft2(ft)
#         image_array = (ift - ift.min()) / (ift.max() - ift.min())
#         np.testing.assert_array_equal(im, (image_array * 255).astype(np.uint8))
                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_band_pass_filter']
    unittest.main()
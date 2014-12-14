'''
Created on Dec 14, 2014

@author: revaz
'''
import unittest
import numpy as np
import image_op.image_io as io
import image_op.image_manip as im


class Test(unittest.TestCase):


    def test_gradient(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        
        mask_x = np.array([[-1,0,1],]*3)
        mask_y = np.transpose(mask_x)
        
        mask_vector = np.array([-1, 0, 1])
        
        gradient_image_x = im.apply_matrix_mask(image, mask_x);
        gradient_image_y = im.apply_matrix_mask(image, mask_y);
        gradient_image_xy = im.apply_array_mask(image, mask_vector, np.transpose(mask_vector))
        
        io.save_array_as_gray_image(gradient_image_x, "../Generated/gradient_x.jpg", normalize=True)
        io.save_array_as_gray_image(gradient_image_y, "../Generated/gradient_y.jpg", normalize=True)
        io.save_array_as_gray_image(gradient_image_xy, "../Generated/gradient_xy.jpg", normalize=True)
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_gradient']
    unittest.main()
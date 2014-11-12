import unittest
from fourier.image_io import read_image
from fourier.image_manip import euclidean_distance
from fourier.image_manip import draw_circle
from fourier.image_manip import create_image

class Test(unittest.TestCase):

    def test_euclidean_distance(self):
        
        self.assertAlmostEqual(euclidean_distance([1, 1], [2, 2]), 1.414, 3)
        self.assertAlmostEqual(euclidean_distance([1,1,1], [2,2,2]), 1.732, 3)
        self.assertRaises(RuntimeError, euclidean_distance, [1,1,1], [22,2])
        
        
    def test_draw_circle(self):
        original_image = read_image('bauckhage.jpg', as_array=True)
        new_image = draw_circle(original_image, 30, 50, True) 
        error_msg = 'Wrong area of the image has been modified'
        center = original_image.shape[0] / 2
        
        # check if the point which is not in the range of radii from the center is unchanged      
        self.assertEquals(original_image[center, center - 51], new_image[center, center - 51], error_msg)
        self.assertEquals(original_image[center, center - 29], new_image[center, center - 29], error_msg)
        self.assertEquals(original_image[center, center + 29], new_image[center, center + 29], error_msg)
        self.assertEquals(original_image[center, center + 51], new_image[center, center + 51], error_msg)
      
        # check if the point which is in the range of radii from the center is set to 0
        self.assertEquals(0, new_image[center, center - 49], error_msg)
        self.assertEquals(0, new_image[center, center - 31], error_msg)
        self.assertEquals(0, new_image[center, center + 31], error_msg)
        self.assertEquals(0, new_image[center, center + 49], error_msg)

    
    def test_create_image(self):
        imA = read_image('bauckhage.jpg', as_array=False)
        imB = read_image('clock.jpg', as_array=False)
        imC = create_image(imA, imB) 
        self.assertAlmostEqual(imC[10, 10], (152.87045191206457+1.4814900926757266e-15j), 5)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
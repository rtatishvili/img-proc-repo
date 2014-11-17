import unittest

from Helper.image_io import read_image
from Helper.image_manip import euclidean_2points
from Helper.image_manip import draw_circle
from Helper.image_manip import combine_magnitude_and_phase


class Test(unittest.TestCase):
    def test_euclidean_distance(self):
        self.assertAlmostEqual(euclidean_2points([1, 1], [2, 2]), 1.414, 3)
        self.assertAlmostEqual(euclidean_2points([1, 1, 1], [2, 2, 2]), 1.732, 3)
        self.assertRaises(RuntimeError, euclidean_2points, [1, 1, 1], [22, 2])

    def test_create_image(self):
        imA = read_image('../Resources/bauckhage.jpg', as_array=False)
        imB = read_image('../Resources/clock.jpg', as_array=False)
        imC = combine_magnitude_and_phase(imA, imB)
        self.assertAlmostEqual(imC[10, 10], (152.87045191206457 + 1.4814900926757266e-15j), 5)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
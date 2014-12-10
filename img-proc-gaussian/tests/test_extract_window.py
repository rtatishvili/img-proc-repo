import unittest
import numpy as np
from image_op.image_manip import extract_window

class Test(unittest.TestCase):
    def test_extract_inner(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (4, 4)
        expected = np.array([[16, 17, 18, 19, 20],
                             [23, 24, 25, 26, 27],
                             [30, 31, 32, 33, 34],
                             [37, 38, 39, 40, 41],
                             [44, 45, 46, 47, 48]])

        window = extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)

    def test_extract_beyond_left(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (4, 0)
        expected = np.array([[0, 0, 14, 15, 16],
                             [0, 0, 21, 22, 23],
                             [0, 0, 28, 29, 30],
                             [0, 0, 35, 36, 37],
                             [0, 0, 42, 43, 44]])

        window = extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)

    def test_extract_beyond_right(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (4, 6)
        expected = np.array([[18, 19, 20, 0, 0 ],
                             [25, 26, 27, 0, 0 ],
                             [32, 33, 34, 0, 0 ],
                             [39, 40, 41, 0, 0 ],
                             [46, 47, 48, 0, 0 ]])

        window = extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)


    def test_extract_beyond_top(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (0, 4)
        expected = np.array([[ 0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0],
                             [ 2,  3,  4,  5,  6],
                             [ 9, 10, 11, 12, 13],
                             [16, 17, 18, 19, 20]])

        window = extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)

    def test_extract_beyond_bottom(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (6, 4)
        expected = np.array([[30, 31, 32, 33, 34],
                             [37, 38, 39, 40, 41],
                             [44, 45, 46, 47, 48],
                             [ 0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0]])

        window = extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)

    def test_extract_beyond_corner(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (0, 0)
        expected = np.array([[ 0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0],
                             [ 0,  0,  0,  1,  2],
                             [ 0,  0,  7,  8,  9],
                             [ 0,  0, 14, 15, 16]])

        window = extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)

    def test_extract_wrong_sizes(self):
        img = np.arange(49).reshape(7, 7)
        center = (0, 0)

        size = (4, 5)
        self.assertRaises(AssertionError, extract_window, img, size, center)
        size = (5, 4)
        self.assertRaises(AssertionError, extract_window, img, size, center)
        size = (4, 4)
        self.assertRaises(AssertionError, extract_window, img, size, center)
        size = (3, 5)
        self.assertRaises(AssertionError, extract_window, img, size, center)

    

    #def test_extract_fill_with_constant(self):
    #    self.fail("Not implemented")
    
    #def test_extract_fill_with_edge(self):
    #    self.fail("Not implemented")

    #def test_extract_fill_with_mirror(self):
    #    self.fail("Not implemented")


if __name__ == '__main__':
    unittest.main()

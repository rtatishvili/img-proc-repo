import unittest

import numpy as np

import image_op.image_manip as im



class Test(unittest.TestCase):


    def test_extract_padding(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (4, 4)
        expected = np.array([[ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  1,  2,  3,  4,  5,  6, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 7,  8,  9, 10, 11, 12, 13, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0,14, 15, 16, 17, 18, 19, 20, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0,21, 22, 23, 24, 25, 26, 27, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0,28, 29, 30, 31, 32, 33, 34, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0,35, 36, 37, 38, 39, 40, 41, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0,42, 43, 44, 45, 46, 47, 48, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0],
                             [ 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0,  0,  0]])

        image_with_padding = im.extract_window_padding(img, size)
                
        np.testing.assert_array_equal(image_with_padding, expected)
        
            

    
    def test_extract_inner(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (4, 4)
        expected = np.array([[16, 17, 18, 19, 20],
                             [23, 24, 25, 26, 27],
                             [30, 31, 32, 33, 34],
                             [37, 38, 39, 40, 41],
                             [44, 45, 46, 47, 48]])

        window = im.extract_window(img, size, center)
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

        window = im.extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)

    def test_extract_beyond_left_fill_edge(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (4, 0)
        expected = np.array([[14, 14, 14, 15, 16],
                             [21, 21, 21, 22, 23],
                             [28, 28, 28, 29, 30],
                             [35, 35, 35, 36, 37],
                             [42, 42, 42, 43, 44]])

        window = im.extract_window(img, size, center, 'edge')
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

        window = im.extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)

    def test_extract_beyond_right_fill_edge(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (4, 6)
        expected = np.array([[18, 19, 20, 20, 20],
                             [25, 26, 27, 27, 27],
                             [32, 33, 34, 34, 34],
                             [39, 40, 41, 41, 41],
                             [46, 47, 48, 48, 48]])

        window = im.extract_window(img, size, center, 'edge')
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

        window = im.extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)


    def test_extract_beyond_top_fill_edge(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (0, 4)
        expected = np.array([[ 2,  3,  4,  5,  6],
                             [ 2,  3,  4,  5,  6],
                             [ 2,  3,  4,  5,  6],
                             [ 9, 10, 11, 12, 13],
                             [16, 17, 18, 19, 20]])

        window = im.extract_window(img, size, center, 'edge')
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

        window = im.extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)


    def test_extract_beyond_bottom_fill_edge(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (6, 4)
        expected = np.array([[30, 31, 32, 33, 34],
                             [37, 38, 39, 40, 41],
                             [44, 45, 46, 47, 48],
                             [44, 45, 46, 47, 48],
                             [44, 45, 46, 47, 48]])
                             
        window = im.extract_window(img, size, center, 'edge')
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

        window = im.extract_window(img, size, center)
        np.testing.assert_array_equal(window, expected)

    def test_extract_beyond_corner_fill_edge(self):
        img = np.arange(49).reshape(7, 7)
        size = (5, 5)
        center = (0, 0)
        expected = np.array([[ 0,  0,  0,  1,  2],
                             [ 0,  0,  0,  1,  2],
                             [ 0,  0,  0,  1,  2],
                             [ 7,  7,  7,  8,  9],
                             [14, 14, 14, 15, 16]])

        window = im.extract_window(img, size, center, 'edge')
        np.testing.assert_array_equal(window, expected)


if __name__ == '__main__':
    unittest.main()

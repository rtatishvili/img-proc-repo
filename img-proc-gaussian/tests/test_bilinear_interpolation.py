import unittest

import calc.bilinear_interpolation as li
import image_op.image_io as io


class MyTestCase(unittest.TestCase):
    def test_something(self):
        image = io.read_image("../Resources/bauckhage.jpg", as_array=True)
        print image[0, 0], image[0, 1], image[1, 0], image[1, 1]
        assert li.resample(image, 0.8, 0.7) == 109.66


if __name__ == '__main__':
    unittest.main()

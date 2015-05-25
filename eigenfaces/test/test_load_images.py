import unittest
import numpy.testing as npt
from src import image_loader, filepath_generator


class Test(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.FOLDERPATH = "train/"

    def test_load_image_loads_face_image(self):
        filepath = self.FOLDERPATH + "face00001.pgm"
        actual = image_loader.load_image(filepath)
        expected = [104, 122, 142, 159, 162, 158, 167]

        npt.assert_array_equal(actual[0:7], expected, 'Image loader loads wrong image.')

    def test_load_range_of_images(self):
        dataset = image_loader.load_images(0, 10)

        npt.assert_equal(dataset[0, 0:7], [104, 122, 142, 159, 162, 158, 167])
        npt.assert_equal(dataset[1, 0:7], [112, 134, 160, 172, 183, 180, 184])
        npt.assert_equal(dataset[8, 0:7], [ 64,  77,  85,  91, 101, 108, 112])
        npt.assert_equal(dataset[9, 0:7], [ 50,  67,  77,  85,  90,  99, 107])

    def test_loaded_images_shape(self):
        dataset = image_loader.load_images(0, 10)
        actual = dataset.shape
        expected = (10, 361)

        npt.assert_equal(actual, expected)

    def test_count_zeros_return_4_for_1_digit_number(self):
        actual = filepath_generator.count_zeros(9)
        expected = 4

        npt.assert_equal(actual, expected)

    def test_count_zeros_return_3_for_2_digit_number(self):
        actual = filepath_generator.count_zeros(10)
        expected = 3

        npt.assert_equal(actual, expected)

    def test_count_zeros_return_2_for_3_digit_number(self):
        actual = filepath_generator.count_zeros(100)
        expected = 2

        npt.assert_equal(actual, expected)

    def test_filepath_counter_return_face00001_for_1(self):
        actual = filepath_generator.get_filepath_for(1)
        expected = self.FOLDERPATH + "face00001.pgm"

        npt.assert_equal(actual, expected)

    def test_filepath_counter_return_face00016_for_16(self):
        actual = filepath_generator.get_filepath_for(16)
        expected = self.FOLDERPATH + "face00016.pgm"

        npt.assert_equal(actual, expected)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_load_image']
    unittest.main()

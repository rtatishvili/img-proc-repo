import unittest
import src.histogram as hist

class Test(unittest.TestCase):


    def test_return_zero_array_for_empty_image(self):
        
        image_array = []
        actual = hist.calc_histogram(image_array)
        expected = [0] * len(actual)

        self.assertEqual(expected, actual, 'Histogram of an empty image should contain only zeroes')


    def test_return_array_of_length_256(self):
        
        image_array = [0, 1, 2]
        actual = hist.calc_histogram(image_array)
        expected = [0] * 256

        self.assertEqual(len(expected), len(actual), 'Histogram should have 256 elements')

 
    def test_count_distinct_values_in_array(self):
 
        image_array = [0, 2, 2, 5, 5, 5]         
        actual = hist.calc_histogram(image_array)         
        expected = [0] * 256
        expected[0] = 1
        expected[2] = 2
        expected[5] = 3
       
        self.assertEqual(expected, actual, 'Histogram should count distinct values in an image')


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testCalcHistogramShouldCountDistinctValuesInArray']
    unittest.main()

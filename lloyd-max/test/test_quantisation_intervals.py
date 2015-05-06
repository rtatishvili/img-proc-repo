import unittest
import numpy as np
import src.histogram as hist
import src.quantisation as q


class Test(unittest.TestCase):


    def test_init_by_default_count_to_8(self):
        
        intervals = q.init_quantisation_intervals()
        expected = 9 # 9 boundaries means 8 intervals
        actual = len(intervals)
        
        np.testing.assert_equal(expected, actual, 'Quantisation intervals should create 8 intervals by default')

    
    def test_init_have_boundaries_at_almost_same_distance(self):

        actual = q.init_quantisation_intervals(3)
        expected = [0, 256 / 3, 256 * 2 / 3, 256]
        
        np.testing.assert_array_equal(expected, actual, 'Quantisation intervals do not have boundaries at almost same distance')


    def test_update_boundaries_as_means_of_adjacent_points(self):
        
        intervals = q.init_quantisation_intervals(4)
        points = [25, 75, 125, 175]
        actual = q.update_quantisation_intervals(intervals, points)        
        expected = [0, 50, 100, 150, 256]
        
        np.testing.assert_array_equal(expected, actual, 'Interval boundaries is not updated as means of adjacent points')
        
        
    def test_update_points_according_to_prob_density(self):
        
        points = [32, 96, 160, 224]
        intervals = [0, 64, 128, 192, 256]
        image_array = [48, 48, 52, 52, 112, 112, 120, 120, 176, 176, 240, 240]
        prob_density = hist.calc_prob_density(image_array)      
        actual = q.update_quantisation_points(intervals, points, prob_density)
        expected = [50.0, 116.0, 176.0, 240.0]
        
        np.testing.assert_allclose(actual, expected, err_msg='Points are not updated according to the probability density of an image')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_intervals']
    unittest.main()
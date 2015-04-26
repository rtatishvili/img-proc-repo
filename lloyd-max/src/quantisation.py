import numpy as np
import histogram as hist

def init_quantisation_intervals(interval_count=8):
    intervals = np.arange(interval_count + 1) * (256 / interval_count)
    intervals[-1] = 256
    
    return intervals

def init_quantisation_points(interval_count=8):
    return np.arange(interval_count) * (256 / interval_count) + (256 / (2 * interval_count))


def update_quantisation_intervals(intervals, points):
    
    updated = np.copy(intervals)
    prev = points[0]
    i = 1
    
    for item in points[1:]:
        updated[i] = (prev + item) / 2
        prev = item
        i += 1
        
    return updated
        

def update_quantisation_points(intervals, points, image_array):
    
    updated = np.copy(points)
    
    for i in range(len(intervals[:-1])):
        prob_density = hist.calc_prob_density(image_array, intervals[i], intervals[i+1])        
        updated[i] = sum ( np.arange(intervals[i], intervals[i+1]) * (prob_density) )
            
    return updated
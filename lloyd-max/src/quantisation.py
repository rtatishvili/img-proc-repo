import numpy as np
from src.histogram import calc_histogram
from src.histogram import calc_prob_density


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
        

def update_quantisation_points(intervals, points, prob_density):
    
    updated = np.zeros_like(points, np.float)
    
    for i in range(len(intervals[:-1])):       
        start = intervals[i]
        end = intervals[i+1]
        if sum ( prob_density[start:end] ) > 0:
            updated[i] = sum ( np.arange(start, end) * prob_density[start:end] ) / sum ( prob_density[start:end] )
            
    return updated


def square_error(intervals, points, prob_density, histogram):
    
    E = 0.0;
        
    for i in range(len(intervals[:-1])):
        
        start = intervals[i]
        end = intervals[i+1]

        diff_sqr = (np.arange(start, end, dtype=np.float) - float(points[i])) ** 2
        
        E += sum(diff_sqr * histogram[start:end] * prob_density[start:end])

    return E



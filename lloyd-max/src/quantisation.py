import numpy as np

def quantisation_intervals(interval_count=8):
    intervals = np.arange(interval_count + 1) * (256 / interval_count)
    intervals[-1] += 1
    
    return intervals

def quantisation_points(interval_count=8):
    return np.arange(interval_count) * (256 / interval_count) + (256 / (2 * interval_count))



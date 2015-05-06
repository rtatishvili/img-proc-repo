import numpy as np

def calc_histogram(image_array):
    
    histogram = np.zeros(256)
    
    for item in image_array:
        histogram[item] += 1
        
    return histogram


def calc_prob_density(image_array):

    histogram = calc_histogram(image_array)
    sum_density = float(sum(histogram))

    prob_density = np.zeros_like(histogram)
    
    if sum_density > 0.0:
        prob_density = np.array(histogram) / sum_density

    return prob_density



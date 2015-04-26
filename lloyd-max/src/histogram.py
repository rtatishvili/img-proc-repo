import numpy as np

def calc_histogram(image_array):
    
    histogram = [0] * 256
    
    for item in image_array:
        histogram[item] += 1
        
    return histogram


def calc_prob_density(image_array, start=0, end=256):

    histogram = calc_histogram(image_array)
    histogram = histogram[start:end]
    
    sum_density = float(sum(histogram))
    prob_density = np.zeros(end - start)
    
    if sum_density > 0.0:
        prob_density = np.array(histogram) / sum_density

    return prob_density



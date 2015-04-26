def calc_histogram(image_array):
    
    histogram = [0] * 256
    
    for item in image_array:
        histogram[item] += 1
        
    return histogram

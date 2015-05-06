from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import histogram as hist
import quantisation as q
from sympy.plotting import intervalmath


def quantize_image(a, b, image, l=8):
    
    img = np.zeros_like(image)
    b_floor = np.floor(np.array(b))
    
    for v in range(len(a) - 1):
        start_arr = image >= a[v]
        end_arr = image < a[v+1] 
        range_arr = start_arr * end_arr    
        img += b_floor[v] * range_arr
    
    return img

def lloyd_max_algorithm(image_path='../resources/bauckhage-gamma-1.png'):
    
    # Prepare image and parameters
    L = 8    
    MAX_ITER = 50
    
    
    image = Image.open(image_path);
    
    image_array = np.array(image.getdata())
            
    intervals = q.init_quantisation_intervals(L)
    points = q.init_quantisation_points(L)
    prob_density = hist.calc_prob_density(image_array)
    histogram = hist.calc_histogram(image_array)
    
    iter = 0
    epsilon = 1.0
    last_err = 99999999.9
    
    while (iter < MAX_ITER):
        intervals = q.update_quantisation_intervals(intervals, points)
        points = q.update_quantisation_points(intervals, points, prob_density)
        err = q.square_error(intervals, points, prob_density, histogram)
        
        print last_err, err
        
        if abs(err - last_err) < epsilon:            
            break
        
        last_err = err
        iter += 1
        
    quant_image = quantize_image(intervals, points, image_array, L)
    
    quant_image = quant_image.reshape(256, 256)
    
    result = Image.fromarray(quant_image.astype(np.uint8))
    result.save('../results/output.bmp')
    
#     figure = plt.figure()
#     
#     sub = figure.add_subplot(111)
#     
#     sub.imshow(quant_image, cmap=plt.get_cmap('gray'))
#     
#     plt.show()
    


if __name__ == '__main__':
    lloyd_max_algorithm()
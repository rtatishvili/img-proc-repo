from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sqrt, ceil
import numpy as np


def dist_values(distances):

    plt.plot(distances)
    plt.xlabel("distances")
#     plt.ylim(10000, 10000000)
    plt.yscale('linear')
    plt.savefig("results/distances.png", format='png')
    plt.show()



def eigenvalues(curve_tuple):
    e = np.vstack(curve_tuple).T

    plt.plot(e)
    plt.xlabel("eigenvalues")
    plt.ylim(1, 100000)
    plt.yscale('log')
    plt.savefig("results/eigenvalues.png", format='png')
    plt.show()


def sum_of_eigenvalues(eigensum, threshold):
    threshold_line = np.ones_like(eigensum) * threshold
    plt.plot(np.vstack((eigensum, threshold_line)).T)
    plt.savefig("results/cutting_point.png")
    plt.show()


def sum_of_eigenvalues_range(eigensum, threshold, cut):
    threshold_line = np.ones_like(eigensum) * threshold
    start = cut - 15
    end = cut + 14
    
    # calculate proportion of eigensum differences on the
    # both sides of the threshold to locate the intersection
    a = abs(eigensum[cut - 1] - threshold)
    b = abs(eigensum[cut] - threshold)    
    intersection = float(cut) - 1.0 + (a / (a + b))
    
    plt.plot(np.vstack((eigensum[:end], threshold_line[:end])).T)
    plt.plot([intersection, intersection], [0, threshold], color='blue', linewidth=1.5, linestyle="--")
    plt.xlim(start, end)
    plt.ylim(eigensum[start],eigensum[end])
    plt.savefig("results/cutting_point_zoomed.png")
    plt.show()

def normalize_image(array):
    array = 255 * ((array - array.min()) / (array.max() - array.min()))
    return array


def images(v, count):
    dim = int(ceil(sqrt(count)))

    figure = plt.figure(facecolor='white', figsize = (dim, dim))   
    grid = gridspec.GridSpec(dim, dim)
    grid.update(wspace=0.1, hspace=0.05)    

    for i in range(count):    
        image = normalize_image(v[i].reshape(19, 19)).astype(np.uint8)
        result = Image.fromarray(image, 'L')
        sub = figure.add_subplot(grid[i])        
        sub.get_xaxis().set_visible(False)
        sub.get_yaxis().set_visible(False)
        
        sub.imshow(result, cmap=plt.get_cmap('gray'))


    plt.savefig("results/eigenfaces.png")
    plt.show()

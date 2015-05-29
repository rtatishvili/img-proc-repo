from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sqrt, ceil
import numpy as np


def dist_values(distances, filename):

    plt.figure()
    plt.plot(distances)
    plt.xlabel("distances")
    plt.yscale('linear')
    plt.savefig("results/" + filename, format='png')


def dist_comparison(distances, filename):

    count = distances.shape[1] / 2

    for i in range(count):
        plt.figure()
        plt.plot(distances[:,i])
        plt.plot(distances[:,i + count], linestyle='--')
        plt.xlabel("distances")
        plt.yscale('linear')
        plt.savefig("results/" + str(i) + '_' + filename, format='png')


def eigenvalues(curve_tuple):
    e = np.vstack(curve_tuple).T

    plt.figure()
    plt.plot(e)
    plt.xlabel("eigenvalues")
    plt.ylim(1, 100000)
    plt.yscale('log')
    plt.savefig("results/eigenvalues.png", format='png')


def sum_of_eigenvalues(eigensum, threshold):
    threshold_line = np.ones_like(eigensum) * threshold

    plt.figure()
    plt.plot(np.vstack((eigensum, threshold_line)).T)
    plt.savefig("results/cutting_point.png")


def sum_of_eigenvalues_range(eigensum, threshold, cut):
    threshold_line = np.ones_like(eigensum) * threshold
    start = cut - 15
    end = cut + 14

    # calculate proportion of eigensum differences on the
    # both sides of the threshold to locate the intersection
    a = abs(eigensum[cut - 1] - threshold)
    b = abs(eigensum[cut] - threshold)
    intersection = float(cut) - 1.0 + (a / (a + b))

    plt.figure()
    plt.plot(np.vstack((eigensum[:end], threshold_line[:end])).T)
    plt.plot([intersection, intersection], [0, threshold], color='blue', linewidth=1.5, linestyle="--")
    plt.xlim(start, end)
    plt.ylim(eigensum[start],eigensum[end])
    plt.savefig("results/cutting_point_zoomed.png")


def normalize_image(array):
    array = 255 * ((array - array.min()) / (array.max() - array.min()))
    return array


def images(image_set, count, filename='eigenfaces.png'):
    dim = int(ceil(sqrt(count)))
    figure = plt.figure(facecolor='white', figsize = (dim, dim))
    grid = gridspec.GridSpec(dim, dim)
    grid.update(wspace=0.1, hspace=0.05)

    for i in range(count):
        image = normalize_image(image_set[i].reshape(19, 19)).astype(np.uint8)
        result = Image.fromarray(image, 'L')
        sub = figure.add_subplot(grid[i])
        sub.get_xaxis().set_visible(False)
        sub.get_yaxis().set_visible(False)

        sub.imshow(result, cmap=plt.get_cmap('gray'))

    plt.savefig("results/" + filename)

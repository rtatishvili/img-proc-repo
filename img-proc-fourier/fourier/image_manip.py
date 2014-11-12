import numpy as np
from math import sqrt
from fourier.Assert import Assert
import fourier.fourier_calc as fourier_calc



def euclidean_distance(x, y):
    """
    This method calculates Euclidean distance between two points. The client must provide array of the coordinate of the points.
    @param x: first point
    @param y: second point
    @return: distance between x and y
    """
    Assert.isTrue(len(x) == len(y), "Arrays have different length")    
    a = 0.
    for i in range(len(x)):
        a += (x[i] - y[i]) ** 2
    
    return sqrt(a)

def draw_circle(picture_array, r_min, r_max, inside):
    """
    This method draw a black circle in a picture_array. Client must provide the min radius and the max radius of a circle, picture_array.
    @param picture_array: on which will the circle will be drawn,
    @param r_min: min radius of a circle
    @param r_max: max radius of a circle
    @param inside: true if the inside of the circle will be black, otherwise the circle will retain the value from the pixels of the picture_array and the rest of the picture_array will be black 
    """
    for i in range(picture_array.shape[0]):
        for j in range(picture_array.shape[1]):
            norm = euclidean_distance([i, j], [picture_array.shape[0] / 2, picture_array.shape[1] / 2])
            is_in_radius = (norm >= r_min and norm <= r_max)
            if inside == False:
                if is_in_radius == False:
                    picture_array[i, j] = 1                   
            else:
                if is_in_radius == True:
                    picture_array[i, j] = 0                       
    return picture_array

def create_image(imageA, imageB):
    """
    This method creates new image by combining the magnitude form **imageA** and phase from **imageB**. Client must provide
    two images with same resolution.
    @param imageA: first image
    @param imageB: second image
    @return: new image
    """
    Assert.isTrue(imageA.size == imageB.size, "Image A and B have different resolution!")
    ftImageA = np.fft.fft2(imageA)
    ftImageB = np.fft.fft2(imageB)
    imageC = np.zeros(shape=imageA.size, dtype = np.complex)
    for i in range(imageC.shape[0]):
        for j in range(imageC.shape[1]):
            magnitude = fourier_calc.magnitude(ftImageA[i, j])
            phase = fourier_calc.phase(ftImageB[i, j])
            imageC[i, j] =  fourier_calc.create_complex_number(magnitude, phase)
    return np.fft.ifft2(imageC)
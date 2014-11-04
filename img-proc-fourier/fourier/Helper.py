from math import sqrt
from fourier.Assert import Assert
from PIL import Image
import numpy as np

class Helper(object):
    
    def __init__(self, params):    
        pass
    
    @staticmethod
    def euclidean_distance(x, y):
        '''
        This method calculates Euclidean distance between two points. The client must provide array of the coordinate of the points.
        @param x: first point
        @param y: second point
        @return: distance between x and y
        '''
        Assert.isTrue(len(x) == len(y), "Arrays have different length")
        
        a = 0.
        for i in range(len(x)):
            a += (x[i] - y[i]) ** 2
        
        return sqrt(a)
    
    @staticmethod
    def read_image(image_location, as_array):
        '''
        Read image from given location. if as_array == True then the method returns the image as array. 
        @param image_location: path of the image
        @param as_array: if true the method returns the image as array 
        '''
        image = Image.open(image_location, 'r')
        if as_array == True:
            data = []
            pix = image.load()
            for y in xrange(image.size[1]):
                data.append([pix[x, y] for x in xrange(image.size[0])])
            return np.array(data)
        return image
    
    @staticmethod
    def create_image(image_array):
        '''
        This method creates image gray from array.
        @param image_array: from which will be the image created.
        @return: gray style image 
        '''
        return Image.fromarray(np.uint8(image_array), 'L')
    
    @staticmethod
    def draw_circle(picture_array, r_min, r_max, inside):
        '''
        This function draw a black circle in a picture_array. Client must provide the min radius and the max radius of a circle, picture_array.
        @param picture_array: on which will the circle will be drawn,
        @param r_min: min radius of a circle
        @param r_max: max radius of a circle
        @param inside: true if the inside of the circle will be black, otherwise the circle will retain the value from the pixels of the picture_array and the rest of the picture_array will be black 
        '''
        for i in range(picture_array.shape[0]):
            for j in range(picture_array.shape[1]):
                norm = Helper.euclidean_distance([i, j], [picture_array.shape[0] / 2, picture_array.shape[1] / 2])
                is_in_radius = (norm >= r_min and norm <= r_max)
                if inside == False:
                    if is_in_radius == False:
                        picture_array[i, j] = 1                   
                else:
                    if is_in_radius == True:
                        picture_array[i, j] = 0                       
        return picture_array
    
    @staticmethod
    def save_image(image_array, location):
        '''
        This method save array as JPG image. Client must provide the array and the location where the image will be saved.
        @param image_array: that is saved
        @param location: where the image is saved
        '''
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        result = Image.fromarray((image_array * 255).astype(np.uint8))
        result.save(location)
        return result, image_array




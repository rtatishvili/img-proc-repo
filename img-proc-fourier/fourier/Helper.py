from math import sqrt
import sys
from fourier.Assert import Assert

class Helper(object):
    
    def __init__(self, params):    
       """
        Constructor
        """
    @staticmethod
    def euclidean_distance(x, y):
        Assert.isTrue(len(x) == len(y), "Arrays have different length")
        
        a = 0.
        for i in range(len(x)):
            a += (x[i] - y[i]) ** 2
        
        return sqrt(a)
    
    @staticmethod
    def draw_circle(picture, r_min, r_max, inside):
        '''
        This function draw a black circle in a picture. Client must provide the min radius and the max radius of a circle, picture.
        @param picture: on which will the circle will be drawn,
        @param r_min: min radius of a circle
        @param r_max: max radius of a circle
        @param inside: true if the inside of the circle will be black, otherwise the circle will retain the value from the pixels of the picture and the rest of the picture will be black 
        '''
        picture1 = picture.copy()
        result = picture1.load()
        for i in range(picture.size[0]):
            for j in range(picture.size[1]):
                norm = Helper.euclidean_distance([i, j], [picture.size[0] / 2, picture.size[1] / 2])
                is_in_radius = (norm >= r_min and norm <= r_max)
                if inside == False:
                    if is_in_radius == False:
                        result[i, j] = 0                   
                else:
                    if is_in_radius == True:
                        result[i, j] = 0                       
        return picture1




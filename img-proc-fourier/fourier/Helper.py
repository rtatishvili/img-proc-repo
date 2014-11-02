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
    def draw_circle(picture, r_min, r_max):
        picture1 = picture.copy()
        result = picture1.load()
        for i in range(picture.size[0]):
            for j in range(picture.size[1]):
                norm = Helper.euclidean_distance([i, j], [picture.size[0] / 2, picture.size[1] / 2])
                if norm >= r_min and norm <= r_max:
                    result[i, j] = 0                
        return picture1




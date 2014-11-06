from math import sqrt
from fourier.Assert import Assert

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
            norm = euclidean_distance([i, j], [picture_array.shape[0] / 2, picture_array.shape[1] / 2])
            is_in_radius = (norm >= r_min and norm <= r_max)
            if inside == False:
                if is_in_radius == False:
                    picture_array[i, j] = 1                   
            else:
                if is_in_radius == True:
                    picture_array[i, j] = 0                       
    return picture_array


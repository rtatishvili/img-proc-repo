from cmath import sqrt
class Helper(object):
    
    
    @staticmethod
    def euclidean_distance(x, y):
        if len(x) != len(y):
            return 
        
        a = 0
        for i in range(len(x)):
            a += (x[i] - y[i])**2
        
        return sqrt(a)




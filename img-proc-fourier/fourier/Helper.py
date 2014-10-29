from cmath import sqrt
class Helper(object):
    
    
    @classmethod
    def distance(cls, px, py, qx, qy):
        x = px - qx
        y = py - qy
       
        return sqrt(x ** 2 + y ** 2)
    
    




'''
Created on Nov 10, 2014

@author: kostadin
'''
import math as math
import numpy as np
from fourier.Assert import Assert

def phase(val):
    '''
     This method extracts the phase of a complex number. Client must provide complex number.
     @param val: complex number from which the phase is extracted
     @return: phase from a complex number   
    '''
    Assert.isTrue(type(val) is complex, "Parameter is not of type complex!")
    return math.atan2(np.imag(val), np.real(val))

def magnitude(val):
    '''
    This method extracts the magnitude of a complex number. Client must provide complex number.
    @param val: complex number from which the phase is extracted
    @return: magnitude of a complex number
    '''
    Assert.isTrue(type(val) is complex, "Parameter is not of type complex!")
    return math.sqrt(np.imag(val) ** 2 + np.real(val) ** 2)